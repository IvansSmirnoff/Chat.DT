#!/usr/bin/env bash
# Bring up Neo4j inside a Colab VM (or any Ubuntu 22.04 box), next to the runner.
#
# Why: the experiment runs in ONE process beside the database -- restore a model's
# graph snapshot, build the prompt context from the live graph, run its questions
# -- so there is no server, no tunnel and no bundle to ship. Colab has no Docker,
# so Neo4j runs from the tarball, as root, bound to localhost only.
#
# Idempotent: a second run in the same session only (re)starts the server.
# Same version as the docker-compose image (neo4j:5-community = 5.26.x LTS), so
# Cypher behaves identically on the laptop and in Colab. No APOC: nothing uses it.
#
# Usage (Colab cell):   !bash scripts/colab/bootstrap.sh
# Env overrides:        NEO4J_VERSION NEO4J_HOME NEO4J_PASSWORD NEO4J_HEAP_MAX NEO4J_PAGECACHE
set -euo pipefail

NEO4J_VERSION="${NEO4J_VERSION:-5.26.31}"
NEO4J_HOME="${NEO4J_HOME:-/opt/neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-password}"
HEAP="${NEO4J_HEAP_MAX:-4g}"
PAGECACHE="${NEO4J_PAGECACHE:-2g}"
TARBALL="neo4j-community-${NEO4J_VERSION}-unix.tar.gz"
URL="https://dist.neo4j.org/${TARBALL}"

log() { echo "[bootstrap] $*"; }

# --- Java 21 (Neo4j 5.26 runs on 17 or 21) -----------------------------------
if ! java -version 2>&1 | grep -qE 'version "(17|21)\.'; then
  log "installing a Java runtime"
  export DEBIAN_FRONTEND=noninteractive
  apt-get -qq update
  apt-get -qq install -y openjdk-21-jre-headless >/dev/null 2>&1 \
    || apt-get -qq install -y openjdk-17-jre-headless >/dev/null
fi
log "$(java -version 2>&1 | head -1)"

# --- Neo4j tarball, sha256-verified -------------------------------------------
if [ ! -x "${NEO4J_HOME}/bin/neo4j" ]; then
  tmp="$(mktemp -d)"
  log "downloading ${TARBALL}"
  curl -fsSL "${URL}" -o "${tmp}/${TARBALL}"
  curl -fsSL "${URL}.sha256" -o "${tmp}/${TARBALL}.sha256"
  expected="$(awk '{print $1}' "${tmp}/${TARBALL}.sha256")"
  actual="$(sha256sum "${tmp}/${TARBALL}" | awk '{print $1}')"
  if [ "${expected}" != "${actual}" ]; then
    echo "[bootstrap] sha256 mismatch for ${TARBALL}: ${actual} != ${expected}" >&2
    exit 1
  fi
  tar -xzf "${tmp}/${TARBALL}" -C "${tmp}"
  mkdir -p "$(dirname "${NEO4J_HOME}")"
  mv "${tmp}/neo4j-community-${NEO4J_VERSION}" "${NEO4J_HOME}"
  rm -rf "${tmp}"
  "${NEO4J_HOME}/bin/neo4j-admin" dbms set-initial-password "${NEO4J_PASSWORD}" >/dev/null
fi

# --- Configuration: memory, localhost only ------------------------------------
conf="${NEO4J_HOME}/conf/neo4j.conf"
set_conf() {
  if grep -qE "^#?${1}=" "${conf}"; then
    sed -i -E "s|^#?${1}=.*|${1}=${2}|" "${conf}"
  else
    echo "${1}=${2}" >> "${conf}"
  fi
}
set_conf server.memory.heap.initial_size "${HEAP}"
set_conf server.memory.heap.max_size "${HEAP}"
set_conf server.memory.pagecache.size "${PAGECACHE}"
set_conf server.default_listen_address 127.0.0.1

# --- Start and wait for Bolt ----------------------------------------------------
if "${NEO4J_HOME}/bin/neo4j" status >/dev/null 2>&1; then
  log "restarting Neo4j to apply the configuration"
  "${NEO4J_HOME}/bin/neo4j" restart >/dev/null
else
  "${NEO4J_HOME}/bin/neo4j" start >/dev/null
fi
for i in $(seq 1 90); do
  if "${NEO4J_HOME}/bin/cypher-shell" -a bolt://127.0.0.1:7687 -u neo4j -p "${NEO4J_PASSWORD}" \
       "RETURN 1" >/dev/null 2>&1; then
    log "Neo4j ${NEO4J_VERSION} up on bolt://127.0.0.1:7687 after ${i}s (heap ${HEAP}, pagecache ${PAGECACHE})"
    exit 0
  fi
  sleep 1
done
echo "[bootstrap] Neo4j did not come up; last log lines:" >&2
tail -n 30 "${NEO4J_HOME}/logs/neo4j.log" >&2 || true
exit 1
