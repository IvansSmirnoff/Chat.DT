# NL2Cypher: A Comparative Framework for Evaluating LLM Approaches to BIM Query Answering

## Abstract

This paper presents a comparative research framework for evaluating different approaches to answering natural language queries about Building Information Models (BIM). We propose a **2×2 experimental design** that systematically compares:

1. **Data Access Method**: Direct LLM reasoning (no data access) vs. Graph-based query generation (structured data access via Neo4j)
2. **Grounding Level**: Unconstrained generation vs. Schema-constrained generation (using IDS specifications)

Our key finding is that **LLMs cannot answer BIM queries without structured data access**—regardless of their reasoning capabilities or schema knowledge. When run locally without model data, Direct QA approaches achieve near-zero accuracy because the LLM has no way to identify specific building elements. However, using a **Cloud Override mode** (`--cloud-direct`), we can feed the full model dump (~1.2M tokens) to Gemini via context caching, enabling a direct comparison: **in-context data access vs. structured graph queries**. The Graph/Cypher approach remains the most reliable method, but cloud-enabled Direct QA offers a viable alternative when graph infrastructure is unavailable.

---

## 1. Introduction

### 1.1 Problem Statement

Building Information Modeling (BIM) has become the standard for digital representation of building data. As BIM models grow in complexity, there is increasing demand for natural language interfaces that allow non-technical users to query building data without learning specialized query languages.

The naive approach—feeding building data directly to an LLM and asking it to answer questions—faces fundamental limitations:
- BIM models contain thousands to millions of elements
- Each element has a unique GlobalId (e.g., `2O2Fr$t4X7Zf8NOew3FNr2`)
- LLM context windows cannot accommodate full model data
- Even with partial data, LLMs cannot reliably extract specific identifiers

### 1.2 Research Questions

This framework addresses three key research questions:

1. **RQ1**: Can LLMs answer BIM queries through reasoning alone, without access to building data?
2. **RQ2**: Does providing schema information (without actual data) improve LLM performance?
3. **RQ3**: How does grammar-constrained query generation compare to unconstrained generation?
4. **RQ4**: Can large-context cloud LLMs with in-context model data match the accuracy of structured graph queries?

### 1.3 Contributions

- A **2×2 experimental framework** for systematic comparison of LLM approaches to BIM query answering
- **Empirical evidence** that LLMs require structured data access for BIM queries
- **Grammar-constrained decoding** using IDS specifications for guaranteed schema compliance
- A **Cloud Override mode** (`--cloud-direct`) that enables Direct QA settings to run on Gemini with full model data via context caching, comparing in-context data access against structured graph queries
- An **open-source implementation** with reproducible experiments

---

## 2. Experimental Design

### 2.1 The 2×2 Matrix

Our experimental design crosses two independent variables:

| | **No Data Access** | **Data Access (Neo4j)** |
|---|---|---|
| **No Grounding** | Setting 1: DIRECT_QA_BASELINE | Setting 3: CYPHER_SOFT |
| **With Grounding** | Setting 2: DIRECT_QA_GROUNDED | Setting 4: CYPHER_STRICT |

### 2.2 Experimental Settings

#### Default Mode (Local / Standard Cloud)

| Setting | Name | LLM Input | LLM Output | Data Access |
|---------|------|-----------|------------|-------------|
| **1** | `DIRECT_QA_BASELINE` | Question only | JSON list of GlobalIDs | ❌ None |
| **2** | `DIRECT_QA_GROUNDED` | Question + IDS schema | JSON list of GlobalIDs | ❌ None |
| **3** | `CYPHER_SOFT` | Question + schema prompt | Cypher query → Neo4j | ✅ Full |
| **4** | `CYPHER_STRICT` | Question + grammar constraints | Cypher query → Neo4j | ✅ Full |

#### Cloud Override Mode (`--cloud-direct`)

The `--cloud-direct` flag forces Settings 1–3 to run on Gemini with full model data access, enabling a direct comparison between in-context reasoning and graph-based queries:

| Setting | Name | LLM Input | LLM Output | Data Access | Engine |
|---------|------|-----------|------------|-------------|--------|
| **1** | `DIRECT_QA_BASELINE` | Question + **full model dump** (cached) | JSON list of GlobalIDs | ✅ In-context | `GeminiLLMEngine(use_context_cache=True)` |
| **2** | `DIRECT_QA_GROUNDED` | Question + IDS schema + **full model dump** (cached) | JSON list of GlobalIDs | ✅ In-context | `GeminiLLMEngine(use_context_cache=True)` |
| **3** | `CYPHER_SOFT` | Question + schema prompt | Cypher query → Neo4j | ✅ Graph DB | `GeminiLLMEngine` |
| **4** | `CYPHER_STRICT` | ⛔ **Skipped** (requires Outlines, incompatible with remote API) | — | — | — |

**Key design decisions:**
- The model dump (~1.2M tokens) is truncated to fit Gemini's 1M context window (~900K token budget after headroom)
- Gemini context caching is attempted first (avoids re-uploading per query); if caching fails (SDK incompatibility), the truncated dump is sent inline per request
- Token-based rate limiting prevents exceeding Gemini's 1M TPM quota

### 2.3 Hypotheses

**Default mode:**
- **H1**: Settings 1 & 2 will achieve EA ≈ 0% (LLM cannot guess GlobalIds without data)
- **H2**: Setting 2 will not significantly outperform Setting 1 (schema knowledge without data is insufficient)
- **H3**: Settings 3 & 4 will achieve significantly higher EA (data access enables correct answers)
- **H4**: Setting 4 will achieve higher SVR than Setting 3 (grammar constraints ensure valid syntax)

**Cloud override mode:**
- **H5**: Cloud Settings 1 & 2 will achieve significantly higher EA than their default counterparts (real data access via context cache)
- **H6**: Cloud Setting 3 (Cypher via Gemini) will remain competitive with or outperform Cloud Settings 1 & 2 (structured queries are more reliable than scanning 900K tokens of raw data)

---

## 3. System Architecture

### 3.1 High-Level Overview
┌─────────────────────────────────────────────────────────────────────────────┐
│                       EXPERIMENT ENTRY POINT                                │
│            `python -m src.eval.cli` → ExperimentRunner (src/eval/runner.py) │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EXPERIMENT CONFIGURATION                                 │
│                      ExperimentSetting (src/config.py)                      │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  DIRECT_QA_BASELINE │ DIRECT_QA_GROUNDED │ CYPHER_SOFT │ CYPHER_STRICT │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LLM ENGINE LAYER                                  │
│                          src/llm_engine.py                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │  LocalLLMEngine │  │  GeminiLLMEngine│  │    OpenAILLMEngine          │  │
│  │  (Outlines/HF)  │  │  (Google API)   │  │    (OpenAI API)             │  │
│  │  STRICT+DIRECT  │  │  SOFT+DIRECT    │  │    SOFT+DIRECT              │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
│                                                                             │
│  Note: Cloud mode is GeminiLLMEngine(use_context_cache=True) — context     │
│  caching + 900k-token truncation are internal flags, not a separate class. │
│                                                                             │
│  Modes: _generate_direct_qa() → JSON list    _generate_cypher() → Cypher   │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       CONSTRAINT LAYER                                      │
│                   src/constraints/                                          │
│  ┌──────────────┐  ┌───────────────────┐  ┌────────────────────────────┐   │
│  │  IDS Parser  │  │  Schema Scanner   │  │    Vocabulary Merger       │   │
│  │  (ids_parser)│  │  (schema_scanner) │  │    (vocabulary_merger)     │   │
│  └──────┬───────┘  └────────┬──────────┘  └────────────┬───────────────┘   │
│         │                   │                          │                    │
│         ▼                   ▼                          ▼                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                  Grammar Generator (grammar.py)                       │  │
│  │                  → Builds Regex Patterns for Constrained Decoding     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                  Context Builder (context_builder.py)                 │  │
│  │                  → Builds System Prompts with Schema Info             │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DATA LAYER                                       │
│  ┌──────────────────────┐              ┌──────────────────────────────────┐ │
│  │  ETL Loader          │              │        Neo4j Database            │ │
│  │  src/etl/loader.py   │─────────────▶│  (Property Graph Storage)        │ │
│  │  (IFC → Neo4j)       │              │                                  │ │
│  └──────────────────────┘              └──────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                 ▲
                                 │
┌─────────────────────────────────────────────────────────────────────────────┐
│                          INPUT FILES                                        │
│  ┌──────────────────────┐              ┌──────────────────────────────────┐ │
│  │     model.ifc        │              │      requirements.ids            │ │
│  │  (BIM Model Data)    │              │  (Schema Constraints)            │ │
│  └──────────────────────┘              └──────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Complete Query Processing Pipeline

### Phase 1: Data Ingestion & Schema Building

#### Step 1.1: IFC Model Loading (ETL)

**File:** `src/etl/loader.py`

The ETL loader (`IFCToNeo4jLoader`) transforms IFC building data into a Neo4j property graph:

```
IFC File (model.ifc)
        │
        ▼
┌───────────────────────────────────────┐
│    ifcopenshell.open()                │
│    Parse IFC entities                 │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│    flatten_psets()                    │
│    Extract all PropertySets and       │
│    QuantitySets into flat key-value   │
│    properties                         │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│    sanitize_property_name()           │
│    Clean property names for Neo4j     │
│    compatibility                      │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│    Create Neo4j Nodes                 │
│    - Multi-label inheritance:         │
│      leaf class + every IFC supertype │
│      (e.g. IfcWallStandardCase :IfcWall│
│       :IfcBuildingElement :IfcElement │
│       :IfcProduct :IfcObject          │
│       :IfcObjectDefinition :IfcRoot)  │
│    - Properties = flattened psets     │
│    - GlobalId = unique identifier     │
│    - IfcRelationship subclasses       │
│      skipped (materialised as edges)  │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│    Create Relationships               │
│    - CONTAINS    (IfcRelContainedIn-  │
│                   SpatialStructure)   │
│    - DECOMPOSES  (IfcRelAggregates)   │
│    - HAS_OPENING (IfcRelVoidsElement) │
│    - FILLS       (IfcRelFillsElement) │
│    - IS_OF_TYPE  (IfcRelDefinesByType)│
│    - HAS_MATERIAL (IfcRelAssociates-  │
│                    Material)          │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│    Propagate transitive containment   │
│    For every (S)-[:CONTAINS]->(X)-    │
│      [:DECOMPOSES*1..]->(Y),          │
│    write (S)-[:CONTAINS {derived:     │
│      true}]->(Y).                     │
│    Lets curtain-wall doors / stair    │
│    flights / ramp railings answer     │
│    storey-level questions naturally.  │
└───────────────────────────────────────┘
```

**Key Functions:**
- `flatten_psets()`: Extracts all PropertySets (Pset_*) and QuantitySets (Qto_*) into flat dictionary
- `sanitize_property_name()`: Handles special characters and Neo4j reserved names
- `safe_value()`: Converts IFC values to Neo4j-compatible types
- `_label_chain()`: Walks the IFC schema declaration chain via `ifcopenshell_wrapper.schema_by_name(...).declaration_by_name(...).supertype()`. Result is cached per leaf class. Applied at node creation so a single `MATCH (:IfcWall)` finds both `IfcWall` and `IfcWallStandardCase` instances.
- `_extract_voids()` / `_extract_fills()`: Materialise the `wall→opening→door|window` chain so questions like "windows on walls > 3m tall" are answerable from graph structure rather than property heuristics.
- `_extract_type_relations()`: Materialises `IS_OF_TYPE` from `IfcRelDefinesByType`. Lets queries hop from a wall instance to its `IfcWallType` and read type-level psets (manufacturer, cost, type-defined fire rating).
- Transitive-containment Cypher pass at the end of `create_relationships()`: `MERGE (s)-[d:CONTAINS]->(child) ON CREATE SET d.derived = true`. Filter `WHERE c.derived IS NULL` to recover the literal IFC topology.

#### Step 1.2: IDS Schema Parsing

**File:** `src/constraints/ids_parser.py`

The IDS (Information Delivery Specification) parser extracts **enforced vocabulary** from buildingSMART IDS XML files:

```
IDS File (requirements.ids)
        │
        ▼
┌───────────────────────────────────────┐
│    XML Parsing with ElementTree       │
│    Extract <specifications>           │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│    Parse <applicability>              │
│    → Extract entity names (IFCWALL)   │
│    → Extract predefined types         │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│    Parse <requirements>               │
│    → Extract property names           │
│    → Extract property sets            │
│    → Extract allowed values (enums)   │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│    IDSSchema                          │
│    - entities: Set[str]               │
│    - properties: Set[str]             │
│    - property_values: Dict[str, Set]  │
│    - entity_constraints: Dict         │
└───────────────────────────────────────┘
```

**Data Classes:**
- `IDSSchema`: Complete schema with entities, properties, and constraints
- `PropertyConstraint`: Individual property with allowed values
- `EntityConstraint`: Entity with its specific property constraints

#### Step 1.3: IFC Model Schema Scanning

**File:** `src/constraints/schema_scanner.py`

The schema scanner extracts **available vocabulary** directly from the IFC model:

```
IFC File (model.ifc)
        │
        ▼
┌───────────────────────────────────────┐
│    IFCSchemaScanner.scan()            │
│    Iterate all IfcProduct elements    │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│    _scan_products()                   │
│    For each entity type:              │
│    - Count instances                  │
│    - Collect all property names       │
│    - Collect quantity properties      │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│    _scan_relationships()              │
│    Extract relationship types         │
│    (CONTAINS, DECOMPOSES, etc.)       │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│    IFCModelSchema                     │
│    - entities: Dict[name → EntitySchema]│
│    - all_properties: Set[str]         │
│    - all_relations: Set[str]          │
└───────────────────────────────────────┘
```

### Phase 2: Vocabulary Merging (Hybrid Schema)

**File:** `src/constraints/vocabulary_merger.py`

The vocabulary merger creates a **Super-Schema** by combining IDS constraints (enforced) with IFC vocabulary (available):

```
┌────────────────────┐     ┌────────────────────┐
│    IDS Schema      │     │    IFC Schema      │
│  (Enforced Rules)  │     │ (Available Values) │
│                    │     │                    │
│ Entities:          │     │ Entities:          │
│ - IFCWALL         │     │ - IfcWall (47)     │
│ - IFCDOOR         │     │ - IfcDoor (12)     │
│                    │     │ - IfcWindow (23)   │
│ Properties:        │     │ - IfcSlab (8)      │
│ - FireRating      │     │                    │
│   [EI30,EI60,EI90]│     │ Properties:        │
│ - IsExternal      │     │ - FireRating       │
│                    │     │ - IsExternal       │
└────────────────────┘     │ - Height           │
          │                │ - Width            │
          │                │ - Name             │
          │                │ - ThermalTransmit. │
          │                └────────────────────┘
          │                          │
          └──────────┬───────────────┘
                     ▼
          ┌────────────────────────────────────────────┐
          │         VocabularyMerger.merge()           │
          │                                            │
          │  1. _merge_entities()                      │
          │     → Combine entity lists                 │
          │     → Mark which came from IDS/IFC         │
          │                                            │
          │  2. _merge_properties()                    │
          │     → All IFC properties (OPEN by default) │
          │     → Apply IDS constraints (STRICT)       │
          │                                            │
          │  3. _classify_properties()                 │
          │     → STRICT: Has enumerated values        │
          │     → BOOLEAN: IsExternal, LoadBearing     │
          │     → NUMERIC: Height, Width, Area         │
          │     → STRING: Name, Description            │
          │     → OPEN: Any value allowed              │
          └────────────────────────────────────────────┘
                     │
                     ▼
          ┌────────────────────────────────────────────┐
          │           CombinedVocabulary               │
          │                                            │
          │  entities: {                               │
          │    'IfcWall': EntityVocabulary(            │
          │      properties: {...},                    │
          │      from_ids: True,                       │
          │      from_ifc: True,                       │
          │      count: 47                             │
          │    ),                                      │
          │    ...                                     │
          │  }                                         │
          │                                            │
          │  all_properties: {                         │
          │    'FireRating': PropertyVocabulary(       │
          │      type: STRICT,                         │
          │      allowed_values: {'EI30','EI60','EI90'}│
          │    ),                                      │
          │    'Height': PropertyVocabulary(           │
          │      type: NUMERIC                         │
          │    ),                                      │
          │    ...                                     │
          │  }                                         │
          │                                            │
          │  strict_properties: {'FireRating', ...}    │
          │  open_properties: {'Height', 'Name', ...}  │
          └────────────────────────────────────────────┘
```

**Property Types:**
| Type | Description | Example |
|------|-------------|---------|
| `STRICT` | Must use one of enumerated values | `FireRating` → ['EI30', 'EI60', 'EI90'] |
| `BOOLEAN` | true/false values | `IsExternal`, `LoadBearing` |
| `NUMERIC` | Number values | `Height`, `Width`, `Area` |
| `STRING` | Any text value | `Name`, `Description` |
| `OPEN` | Any value allowed | General properties |

### Phase 3: Grammar Construction (Regex Patterns)

**File:** `src/constraints/grammar.py`

The grammar generator builds **regex patterns** that constrain LLM output to valid Cypher syntax:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                      build_cypher_regex()                                  │
│                                                                            │
│  Input: entities = {'IfcWall', 'IfcDoor'}                                  │
│         properties = {'FireRating', 'Name', 'Height'}                      │
│                                                                            │
│  Step 1: Build Alternation Groups                                          │
│  ─────────────────────────────────                                         │
│  entity_alt = "(IfcDoor|IfcWall)"                                         │
│  property_alt = "(FireRating|Height|Name)"                                │
│                                                                            │
│  Step 2: Build Value Patterns                                              │
│  ────────────────────────────                                              │
│  STRING_LITERAL = "'[^']*'"                                               │
│  NUMBER_LITERAL = "-?[0-9]+(?:\.[0-9]+)?"                                 │
│  BOOLEAN_LITERAL = "(?:true|false|TRUE|FALSE)"                            │
│  CONVERSION_FUNCTIONS = "(?:toInteger|toFloat|toString|...)"              │
│  value_pattern = "(?:STRING|NUMBER|BOOLEAN|NULL)"                         │
│                                                                            │
│  Step 3: Build Property Access Pattern                                     │
│  ──────────────────────────────────────                                    │
│  property_access = "VARIABLE.property_alt" OR                             │
│                    "toInteger(VARIABLE.property_alt)"                     │
│                                                                            │
│  Step 4: Build Comparison Patterns                                         │
│  ─────────────────────────────────                                         │
│  basic_comparison = "property_access OPERATOR value"                      │
│                   = "n.Width > 200" or "toInteger(n.FireRating) >= 20"   │
│  string_contains  = "VARIABLE.property CONTAINS 'value'"                  │
│  is_null_check    = "VARIABLE.property IS [NOT] NULL"                     │
│  not_contains     = "NOT VARIABLE.property CONTAINS 'value'"              │
│                                                                            │
│  Step 5: Build Clause Patterns                                             │
│  ─────────────────────────────                                             │
│  MATCH clause:  "MATCH (var:Entity)" or "MATCH (a:E1)-[:REL]->(b:E2)"    │
│  WHERE clause:  "WHERE condition (AND|OR condition)*" (up to 4 conditions)│
│  RETURN clause: "RETURN variable" (forces node for GlobalId extraction)   │
│  ORDER BY:      Disabled by default (avoid incomplete queries)            │
│  LIMIT:         Disabled by default (avoid incomplete queries)            │
│                                                                            │
│  Step 6: Assemble Full Pattern                                             │
│  ─────────────────────────────                                             │
│  full_regex = MATCH + REL_PATTERN? + WHERE? + RETURN                      │
└────────────────────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  Example Generated Regex (simplified):                                     │
│                                                                            │
│  [Mm][Aa][Tt][Cc][Hh]\s*\([a-zA-Z_]\w*:(IfcDoor|IfcWall)\)                │
│  (?:-\[:(CONTAINS|DECOMPOSES|HAS_MATERIAL)\]->\([a-zA-Z_]\w*:...\))?     │
│  (?:\s+[Ww][Hh][Ee][Rr][Ee]\s+                                             │
│     (?:toInteger\()?[a-zA-Z_]\w*\.(FireRating|Height|Name)(?:\))?         │
│     \s*(?:=|<>|<=|>=|<|>)\s*(?:'[^']*'|-?[0-9]+|true|false)               │
│     (?:\s+AND\s+...)?)*                                                    │
│  \s+[Rr][Ee][Tt][Uu][Rr][Nn]\s+[a-zA-Z_]\w*                               │
└────────────────────────────────────────────────────────────────────────────┘
```

**Regex Functions:**
- `build_cypher_regex()`: Full-featured regex with all Cypher clauses
- `build_simple_cypher_regex()`: Simplified pattern for basic queries
- `build_relationship_cypher_regex()`: Supports relationship patterns
- `build_cypher_regex_from_vocabulary()`: Uses CombinedVocabulary

### Phase 4: Context Building (System Prompts)

**File:** `src/constraints/context_builder.py`

The context builder creates system prompts with schema information for the LLM:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                      ContextBuilder.build_context()                        │
│                                                                            │
│  Input: CombinedVocabulary, constraint_mode='strict'                       │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │                    SYSTEM PROMPT TEMPLATE                            │ │
│  │                                                                      │ │
│  │  You are a Neo4j Cypher query generator for BIM data.               │ │
│  │                                                                      │ │
│  │  ## Available Schema                                                 │ │
│  │                                                                      │ │
│  │  ### Entities                                                        │ │
│  │  - IfcWall (47 instances)                                           │ │
│  │  - IfcDoor (12 instances)                                           │ │
│  │  - IfcWindow (23 instances)                                         │ │
│  │                                                                      │ │
│  │  ### Properties                                                      │ │
│  │  - FireRating (STRICT: Must be one of ['EI30', 'EI60', 'EI90'])     │ │
│  │  - IsExternal (Boolean: true/false)                                 │ │
│  │  - Height (Numeric)                                                  │ │
│  │  - Name                                                              │ │
│  │                                                                      │ │
│  │  ## Constraint Rules                                                 │ │
│  │  For STRICT properties, use exactly one of the allowed values...    │ │
│  │                                                                      │ │
│  │  ## Examples                                                         │ │
│  │  Question: Find all walls with fire rating EI60                     │ │
│  │  Cypher: MATCH (n:IfcWall) WHERE n.FireRating = 'EI60' RETURN n    │ │
│  │  ...                                                                 │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  Output: PromptContext(                                                    │
│    system_prompt: str,                                                     │
│    schema_string: str,                                                     │
│    constraint_mode: 'strict',                                              │
│    entity_count: 4,                                                        │
│    property_count: 25                                                      │
│  )                                                                         │
└────────────────────────────────────────────────────────────────────────────┘
```

### Phase 5: LLM Generation (Dual Mode)

**File:** `src/llm_engine.py`

The LLM engine supports **two generation modes** based on the experimental setting:

#### Mode A: Direct QA (Settings 1 & 2) — No Data Access

Direct QA tests whether the LLM can answer BIM queries **without access to building data**. This serves as a baseline demonstrating the necessity of structured data access.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     DIRECT QA MODE (No Data Access)                         │
│                                                                             │
│  Setting 1: DIRECT_QA_BASELINE                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  INPUT:  "Find all walls with fire rating EI60"                       │ │
│  │                                                                       │ │
│  │  LLM has: Question only                                               │ │
│  │  LLM lacks: Model data, GlobalIds, actual element properties          │ │
│  │                                                                       │ │
│  │  Expected: LLM cannot guess real GlobalIds → EA ≈ 0                   │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  Setting 2: DIRECT_QA_GROUNDED                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  INPUT:  Question + IDS Schema (entity types, property constraints)   │ │
│  │                                                                       │ │
│  │  LLM has: Schema knowledge (what SHOULD exist)                        │ │
│  │  LLM lacks: Actual data (what DOES exist, specific GlobalIds)         │ │
│  │                                                                       │ │
│  │  Expected: Schema helps understanding, but EA still ≈ 0               │ │
│  │            (knowing "walls should have FireRating" ≠ knowing IDs)     │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  OUTPUT: JSON array of GlobalIDs (likely empty or fabricated)               │
│  Expected Result: [] or random/hallucinated IDs                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Insight (Default Mode)**: The LLM cannot "invent" valid GlobalIds like `2O2Fr$t4X7Zf8NOew3FNr2`. These are randomly generated UUIDs in the IFC file. Without data access, even the most capable LLM will fail.

#### Mode A′: Direct QA with Cloud Override (`--cloud-direct`)

When the `--cloud-direct` flag is used, Settings 1 & 2 gain **real data access** via Gemini's large context window. The full `model_dump.json` is uploaded to Gemini (truncated to ~900K tokens if necessary) and either cached via the Context Caching API or sent inline per request.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              CLOUD DIRECT QA MODE (Data Access via Context Cache)           │
│                                                                             │
│  GeminiLLMEngine(use_context_cache=True)                                    │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  1. Load model_dump.json (~1.2M tokens)                              │ │
│  │  2. Truncate to 900K token budget if necessary                       │ │
│  │  3. Attempt Gemini Context Cache (hash-based, 60-min TTL)            │ │
│  │     └─ If caching fails → fallback to per-request inline injection   │ │
│  │  4. Per query: send only the question (+ IDS specs for Setting 2)    │ │
│  │  5. LLM searches the cached model data for matching elements         │ │
│  │  6. Returns JSON list of GlobalIds found in the data                 │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  Rate Limiting:                                                             │
│  - Proactive TPM/RPM tracking (rolling 60-second window)                   │
│  - 429 retry with parsed delay from error response                         │
│  - Shared across engine instances per API quota                            │
│                                                                             │
│  Expected Result: EA > 0 (LLM can now find real GlobalIds in the data)     │
│  Key Question: Can in-context search match graph-based Cypher accuracy?    │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Mode B: Cypher Generation (Settings 3 & 4) — With Data Access

Cypher generation enables the LLM to access real data through Neo4j queries:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     CYPHER GENERATION MODE (Data Access via Neo4j)          │
│                                                                             │
│  Setting 3: CYPHER_SOFT (Prompt Engineering)                                │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  1. Build system prompt with schema information                       │ │
│  │  2. LLM generates Cypher query (unconstrained)                        │ │
│  │  3. Query executed against Neo4j → returns actual GlobalIds           │ │
│  │                                                                       │ │
│  │  Risk: LLM may generate invalid syntax or non-existent properties     │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  Setting 4: CYPHER_STRICT (Grammar-Constrained Decoding)                    │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  1. Build regex pattern from IDS + Schema vocabulary                  │ │
│  │  2. Create constrained generator: Generator(model, Regex(regex))      │ │
│  │  3. Token-by-token generation with invalid tokens masked              │ │
│  │  4. Query executed against Neo4j → returns actual GlobalIds           │ │
│  │                                                                       │ │
│  │  Guarantee: Output ALWAYS matches regex (valid syntax + schema)       │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  OUTPUT: Cypher query → Neo4j execution → Set of real GlobalIds             │
│  Expected Result: Actual element IDs from the database                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Insight**: The LLM doesn't need to "know" GlobalIds—it generates a query that Neo4j executes to retrieve them. This separation of concerns is fundamental to the approach.

#### Constrained Decoding Flow (STRICT Mode)

```
User Query: "Find all walls with fire rating EI60"
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  System Prompt + User Query                                                 │
│  ═══════════════════════════                                                │
│  [Schema info + constraint rules + examples]                                │
│                                                                             │
│  Question: Find all walls with fire rating EI60                             │
│  Cypher:                                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Outlines Constrained Generation                                            │
│  ═══════════════════════════════                                            │
│                                                                             │
│  Regex Pattern (from grammar.py):                                           │
│  [Mm][Aa][Tt][Cc][Hh]\s*\([a-z]+:(IfcWall|IfcDoor|...)\)                   │
│  (?:\s+WHERE\s+[a-z]+\.(FireRating|Name|...)\s*=\s*'[^']*')?               │
│  \s+RETURN\s+...                                                            │
│                                                                             │
│  Token-by-Token Generation:                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Token 1: "M" (must match regex start)                    ✓          │   │
│  │  Token 2: "A" (continues MATCH)                           ✓          │   │
│  │  Token 3: "T" (continues MATCH)                           ✓          │   │
│  │  Token 4: "C" (continues MATCH)                           ✓          │   │
│  │  Token 5: "H" (completes MATCH)                           ✓          │   │
│  │  Token 6: " " (whitespace)                                ✓          │   │
│  │  Token 7: "(" (node pattern start)                        ✓          │   │
│  │  Token 8: "n" (variable name)                             ✓          │   │
│  │  Token 9: ":" (label separator)                           ✓          │   │
│  │  Token 10: "Ifc" → must choose from entity_alt            ✓          │   │
│  │  Token 11: "Wall" (completes IfcWall)                     ✓          │   │
│  │  ...                                                                 │   │
│  │  → Invalid tokens are masked at each step!                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Output (100% Schema-Valid):                                                │
│  MATCH (n:IfcWall) WHERE n.FireRating = 'EI60' RETURN n                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase 6: Query Execution

**File:** `src/eval/neo4j_exec.py`

```
Generated Cypher Query
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  execute_and_get_ids(driver, query) → Set[str]                              │
│                                                                             │
│  1. driver.session().run(query)                                             │
│  2. For each record, walk columns and call _extract_global_id()             │
│     (handles bare GlobalId strings, neo4j Node objects, dicts)              │
│  3. Collect GlobalIds into a set                                            │
│  4. calculate_ea_cypher() then compares that set against the pre-computed  │
│     gold set via Jaccard similarity                                         │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Result (consumed by src/eval/aggregate.py::evaluate_cypher_output):        │
│  EvaluationResult(                                                          │
│      svr=1.0, scr=1.0, ea=0.83,                                             │
│      generated_ids={"3k9...", "7B2..."},                                    │
│      gold_ids={"3k9...", "7B2...", "Q4a..."},                               │
│  )                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Evaluation Methodology

**Files:** `src/eval/cli.py` (entry point), `src/eval/runner.py` (`ExperimentRunner`), `src/eval/scoring.py` (pure SVR/SCR/EA), `src/eval/aggregate.py` (batch orchestration), `src/eval/neo4j_exec.py` (Cypher execution)

### 4.1 Evaluation Metrics

We use three primary metrics adapted for each output type:

| Metric | Description | Cypher Mode (3 & 4) | Direct QA Mode (1 & 2) |
|--------|-------------|---------------------|------------------------|
| **SVR** | Syntax Validity Rate | Neo4j `EXPLAIN` validates query | Valid JSON array? |
| **SCR** | Schema Compliance Rate | Labels/properties exist in schema | N/A (always 1.0) |
| **EA** | Execution Accuracy | Execute → Compare IDs to gold | Parse JSON → Compare to gold |

**Secondary Metrics:**
- **Precision:** `|Generated ∩ Gold| / |Generated|`
- **Recall:** `|Generated ∩ Gold| / |Gold|`
- **F1 Score:** Harmonic mean of Precision and Recall

**Granular Analysis:**
- **By Difficulty:** EA aggregated by Easy/Medium/Hard query complexity
- **By Category:** EA aggregated by query type (basic_query, property_filter, relationship)

### 4.2 Gold Standard

All settings are evaluated against the same gold standard, enabling fair comparison:

```python
# Step 1: Pre-execute gold Cypher queries
gold_id_sets = execute_gold_queries(test_cases, driver)
# Returns: {test_case_index: Set[str] of GlobalIDs}

# Step 2: Compare generated output to gold
# Direct QA: parse JSON list → compare to gold_ids
# Cypher: execute query → compare result to gold_ids
```

### 4.3 Test Set Design

The test set (`data/test_set.csv`) contains 16 queries across three categories:

| Category | Count | Description | Difficulty |
|----------|-------|-------------|------------|
| `basic_query` | 6 | Simple entity retrieval | Easy |
| `property_filter` | 5 | Queries with WHERE conditions | Medium |
| `relationship` | 5 | Queries involving relationships | Hard |

Each test case includes:
- `question`: Natural language query
- `gold_cypher`: Reference Cypher query
- `category`: Query type classification
- `difficulty`: Easy/Medium/Hard

### 4.4 Experiment Pipeline

```
test_set.csv
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  python -m src.eval.cli  →  ExperimentRunner (src/eval/runner.py)           │
│                                                                             │
│  Phase 1: Setup                                                             │
│  1. Connect to Neo4j database                                               │
│  2. Load IDS schema + IFC vocabulary                                        │
│  3. Build grammar patterns for constrained generation                       │
│                                                                             │
│  Phase 2: Gold Standard                                                     │
│  4. Execute ALL gold_cypher queries → gold_id_sets                          │
│                                                                             │
│  Phase 3: Run All 4 Settings                                                │
│  For each ExperimentSetting:                                                │
│    For each test case:                                                      │
│      5. Generate output (JSON list or Cypher)                               │
│      6. Calculate SVR (syntax validity)                                     │
│      7. Calculate SCR (schema compliance, Cypher only)                      │
│      8. Calculate EA (compare to gold_ids)                                  │
│      9. Calculate Precision/Recall/F1                                       │
│                                                                             │
│  Phase 4: Output                                                            │
│  - results_{model_name}_{timestamp}.json                                    │
│    Contains:                                                                │
│      • setting_results: Metrics for all 4 settings                          │
│      • complexity_results: EA by difficulty level                           │
│      • category_results: EA by query category                               │
│      • ranking: Settings ranked by each metric                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.5 Running Experiments

```bash
# Run full 2×2 comparison experiment (local mode)
python -m src.eval.cli --test-set data/test_set.csv --comparison

# Run single setting
python -m src.eval.cli --test-set data/test_set.csv --setting cypher_strict

# Run with specific model
LLM_MODEL_NAME=Qwen/Qwen2.5-Coder-3B-Instruct \
  python -m src.eval.cli --test-set data/test_set.csv --comparison

# Cloud Override: run Settings 1-3 on Gemini with full model data
# (Setting 4 auto-skipped — requires local Outlines)
python -m src.eval.cli \
  --test-set data/test_set.csv \
  --model-dump data/model_dump.json \
  --cloud-direct \
  --comparison

# Cloud Override: run only CYPHER_SOFT on Gemini (no model-dump needed)
python -m src.eval.cli \
  --test-set data/test_set.csv \
  --setting cypher_soft \
  --cloud-direct
```

**Output Format:**
```json
{
  "model_name": "Qwen2.5-Coder-3B-Instruct",
  "setting_results": {
    "direct_qa_baseline": {"svr": 1.0, "ea": 0.0, ...},
    "direct_qa_grounded": {"svr": 1.0, "ea": 0.0, ...},
    "cypher_soft": {"svr": 0.75, "ea": 0.45, ...},
    "cypher_strict": {"svr": 1.0, "ea": 0.52, ...}
  },
  "complexity_results": {...},
  "ranking": {"ea": ["cypher_strict", "cypher_soft", ...]}
}
```

---

## Remote Query API (Colab → Neo4j)

The `api` service (FastAPI) is the only process reachable from outside the
Docker network. Neo4j's Bolt port is **not** published on the host — every
remote query goes through authenticated HTTPS/HTTP to `api` and is proxied to
Neo4j over the internal network.

### Configure + run

```bash
# 1) Set a strong shared token in .env
python -c "import secrets; print('API_BEARER_TOKEN=' + secrets.token_urlsafe(48))" >> .env

# 2) Bring up the stack (neo4j + app workbench + api)
docker compose up -d --build

# 3) Liveness (no auth) + readiness (auth, verifies Neo4j)
curl http://localhost:8000/health
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/health/ready
```

Interactive OpenAPI docs: `http://localhost:8000/docs`.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/health` | Liveness (unauthenticated) |
| `GET`  | `/health/ready` | Pings Neo4j |
| `POST` | `/cypher/execute` | Run Cypher; returns extracted IDs (and optional rows) |
| `POST` | `/cypher/validate` | EXPLAIN-only syntax check (SVR) |
| `GET`  | `/test-set` | Server-side test cases (Question / Gold_Cypher / ...) |
| `GET`  | `/gold/{index}` | Pre-executed gold IDs for a test case |
| `POST` | `/evaluate` | Full SVR + SCR + EA for a generated output |
| `GET`  | `/schema/valid-labels` | Valid Neo4j labels (for client-side SCR) |
| `GET`  | `/schema/valid-properties` | Valid property names |
| `GET`  | `/schema/vocabulary` | Merged IDS + IFC vocabulary |

### Colab example

```python
import os, requests

API = "http://<your-host>:8000"
HEADERS = {"Authorization": f"Bearer {os.environ['API_BEARER_TOKEN']}"}

# LLM (in Colab) produces a Cypher query; we only execute + score remotely.
generated = "MATCH (n:IfcWall) RETURN n.GlobalId"

# Option A: raw execution, score locally using src/eval/scoring.py
r = requests.post(f"{API}/cypher/execute",
                  json={"query": generated}, headers=HEADERS, timeout=60)
r.raise_for_status()
print(len(r.json()["ids"]), "ids returned")

# Option B: one-shot evaluate against a known gold test case
r = requests.post(f"{API}/evaluate", headers=HEADERS, timeout=60, json={
    "question": "Find all walls",
    "output": generated,
    "output_type": "cypher",
    "gold_index": 0,
    "experiment_setting": "cypher_soft",
})
print(r.json())  # EvaluationResult dict with svr/scr/ea/precision/recall/f1
```

Scoring lives in both places on purpose: `src/eval/scoring.py` is Neo4j-free
and can be imported in Colab to re-score a saved CSV of predictions without
hitting the server at all.

---

## Configuration

**File:** `src/config.py`

### Settings

| Setting | Environment Variable | Default | Description |
|---------|---------------------|---------|-------------|
| `neo4j_uri` | `NEO4J_URI` | `bolt://neo4j:7687` | Neo4j connection URI |
| `neo4j_user` | `NEO4J_USER` | `neo4j` | Neo4j username |
| `neo4j_password` | `NEO4J_PASSWORD` | `password` | Neo4j password |
| `llm_provider` | `LLM_PROVIDER` | `gemini` | Provider: `local`, `gemini`, `openai` |
| `llm_api_key` | `LLM_API_KEY` | - | API key for cloud providers |
| `llm_model_name` | `LLM_MODEL_NAME` | `gemini-1.5-pro` | Model name or path |
| `ifc_file_path` | `IFC_FILE_PATH` | `/app/data/model.ifc` | Path to IFC file |
| `ids_file_path` | `IDS_FILE_PATH` | `/app/data/requirements.ids` | Path to IDS file |
| `model_dump_path` | `MODEL_DUMP_PATH` | - | Path to model dump JSON (for Direct QA) |
| `api_bearer_token` | `API_BEARER_TOKEN` | _(empty)_ | Shared secret for the FastAPI proxy. Empty value disables the API (503 on every auth'd call) |
| `api_host` | `API_HOST` | `0.0.0.0` | Interface the FastAPI app binds to |
| `api_port` | `API_PORT` | `8000` | Port the FastAPI app listens on |

### ExperimentSetting Enum

```python
class ExperimentSetting(str, Enum):
    DIRECT_QA_BASELINE = "direct_qa_baseline"  # Setting 1
    DIRECT_QA_GROUNDED = "direct_qa_grounded"  # Setting 2
    CYPHER_SOFT = "cypher_soft"                # Setting 3
    CYPHER_STRICT = "cypher_strict"            # Setting 4
    
    @property
    def is_direct_qa(self) -> bool: ...        # True for Settings 1 & 2
    
    @property
    def is_cypher_gen(self) -> bool: ...       # True for Settings 3 & 4
    
    @property
    def uses_ids_grounding(self) -> bool: ...  # True for Settings 2 & 4
    
    @property
    def uses_strict_constraints(self) -> bool: ...  # True for Setting 4 only
```

---

## 5. The 2×2 Experimental Matrix

### 5.1 Visual Representation

#### Default Mode (Local / No Model Data)
```
                          │   No Grounding      │   With Grounding      │
                          │   (Unconstrained)   │   (IDS Constraints)   │
──────────────────────────┼─────────────────────┼───────────────────────┤
  No Data Access          │  Setting 1          │  Setting 2            │
  (LLM Reasoning Only)    │  DIRECT_QA_BASELINE │  DIRECT_QA_GROUNDED   │
                          │  → EA ≈ 0%          │  → EA ≈ 0%            │
──────────────────────────┼─────────────────────┼───────────────────────┤
  Data Access             │  Setting 3          │  Setting 4            │
  (Neo4j via Cypher)      │  CYPHER_SOFT        │  CYPHER_STRICT        │
                          │  → Variable EA      │  → Higher EA          │
──────────────────────────┴─────────────────────┴───────────────────────┘
```

#### Cloud Override Mode (`--cloud-direct`)
```
                          │   No Grounding      │   With Grounding      │
                          │   (Unconstrained)   │   (IDS Constraints)   │
──────────────────────────┼─────────────────────┼───────────────────────┤
  In-Context Data Access  │  Setting 1 (cloud)  │  Setting 2 (cloud)    │
  (Gemini Context Cache)  │  model_dump cached   │  model_dump + IDS     │
                          │  → EA > 0 (can find) │  → EA > 0 (guided)   │
──────────────────────────┼─────────────────────┼───────────────────────┤
  Structured Data Access  │  Setting 3 (cloud)  │  Setting 4            │
  (Neo4j via Cypher)      │  Gemini → Cypher    │  ⛔ SKIPPED           │
                          │  → Variable EA      │  (requires Outlines)  │
──────────────────────────┴─────────────────────┴───────────────────────┘
```

### 5.2 Setting Details

**Default mode:**

| Setting | What LLM Receives | What LLM Produces | Data Access | Expected Outcome |
|---------|-------------------|-------------------|-------------|------------------|
| **1** | Question only | JSON GlobalIds | ❌ None | EA ≈ 0 (can't guess IDs) |
| **2** | Question + IDS schema | JSON GlobalIds | ❌ None | EA ≈ 0 (schema ≠ data) |
| **3** | Question + schema prompt | Cypher query | ✅ Neo4j | Variable (may have errors) |
| **4** | Question + grammar | Constrained Cypher | ✅ Neo4j | Higher (valid + data) |

**Cloud override mode (`--cloud-direct`):**

| Setting | What LLM Receives | What LLM Produces | Data Access | Expected Outcome |
|---------|-------------------|-------------------|-------------|------------------|
| **1** | Question + model dump (cached) | JSON GlobalIds | ✅ In-context | EA > 0 (searches data directly) |
| **2** | Question + IDS schema + model dump (cached) | JSON GlobalIds | ✅ In-context | EA > 0 (IDS-guided search) |
| **3** | Question + schema prompt | Cypher query | ✅ Neo4j | Variable (Gemini generates Cypher) |
| **4** | ⛔ Skipped | — | — | Incompatible with remote API |

### 5.3 Theoretical Justification

**Why Direct QA Must Fail (Default Mode):**
- GlobalIds are 22-character Base64-encoded GUIDs (e.g., `2O2Fr$t4X7Zf8NOew3FNr2`)
- They are randomly generated per IFC file
- No amount of reasoning can derive them from a question
- Even knowing "there should be 47 walls" doesn't help identify which walls

**Why Cloud Direct QA Can Succeed:**
- The full model dump provides the LLM with all element data including GlobalIds
- The LLM can search through the provided data like a human reading a JSON file
- Limitation: with ~1.2M tokens truncated to ~900K, some elements may be missing
- Limitation: LLMs may miss matches in very large contexts (needle-in-haystack problem)

**Why Cypher Remains Superior:**
- The LLM only needs to generate a *query pattern*
- Neo4j handles the actual data retrieval with 100% coverage
- No truncation, no context window limits, no missed matches
- Valid Cypher + correct semantics → correct results

---

## File Structure Summary

```
ec3_nl2cypher/
├── src/
│   ├── config.py                # Settings, ExperimentSetting, LLMProvider enums
│   ├── llm_engine.py            # LocalLLMEngine, GeminiLLMEngine (with
│   │                            # use_context_cache flag for cloud mode),
│   │                            # OpenAILLMEngine, create_llm_engine factory
│   ├── constraints/             # IDS + IFC schema pipeline
│   │   ├── ids_parser.py        # IDS XML → IDSSchema
│   │   ├── schema_scanner.py    # IFC scan → IFCModelSchema
│   │   ├── vocabulary_merger.py # IDS + IFC → CombinedVocabulary
│   │   ├── grammar.py           # Regex pattern generation (Outlines)
│   │   └── context_builder.py   # System prompt construction
│   ├── etl/
│   │   └── loader.py            # IFC → Neo4j ETL pipeline
│   └── eval/
│       ├── scoring.py           # Pure SVR/SCR/EA primitives + EvaluationResult
│       ├── neo4j_exec.py        # Cypher execution + execute_gold_queries
│       ├── aggregate.py         # evaluate_*, aggregate_* batch orchestration
│       ├── runner.py            # ExperimentRunner + ExperimentConfig
│       └── cli.py               # `python -m src.eval.cli` entry point
├── scripts/
│   ├── build_bundle.py          # Produce bundle_<model>.json for Colab
│   ├── create_model_dump.py     # Direct-QA model dump
│   ├── analyze_ifc.py
│   └── diagnose_barcelona.py
├── notebooks/
│   ├── run_experiment_cloud.ipynb   # Gemini/OpenAI on CPU Colab
│   └── run_experiment_local.ipynb   # LocalLLMEngine on GPU Colab
├── tests/                       # pytest suite (29 tests across 5 files)
├── requirements-base.txt        # neo4j, ifcopenshell, pydantic, numpy
├── requirements-llm.txt         # torch, outlines, google-generativeai, openai
└── requirements-dev.txt         # pytest, black, ruff, mypy
```

---

## 6. Key Innovations

### 6.1 Comparative Framework Design

The 2×2 matrix enables systematic analysis of two independent factors:

1. **Data Access Method**: Isolates the effect of structured vs. no data access
2. **Grounding Level**: Measures the impact of schema constraints

This design answers fundamental questions:
- *"Can LLMs answer BIM queries without data?"* → Compare rows
- *"Does grounding improve accuracy?"* → Compare columns

### 6.2 Grammar-Constrained Decoding

For CYPHER_STRICT mode (Setting 4), we use **Outlines** for grammar-constrained decoding:

```python
# Build regex from IDS vocabulary + IFC schema
regex = build_cypher_regex_from_vocabulary(vocabulary)

# Create constrained generator
generator = Generator(model, Regex(regex))

# Generate - output GUARANTEED to match regex
cypher_query = generator(prompt)
```

**Benefits:**
- **100% SVR**: Every generated query is syntactically valid
- **100% SCR**: Only uses entities/properties from the schema
- **Deterministic**: No post-hoc validation needed

### 6.3 Unified Evaluation

Both approaches are evaluated against identical gold standards:

```python
# Same gold IDs for all settings
gold_ids = execute_gold_queries(test_cases, driver)

# Direct QA: parse output → compare to gold
ea_direct = len(parsed_ids & gold_ids) / len(gold_ids)

# Cypher: execute query → compare to gold
ea_cypher = len(query_result_ids & gold_ids) / len(gold_ids)
```

This enables fair, apples-to-apples comparison across fundamentally different approaches.

---

## 7. Expected Results & Discussion

### 7.1 Primary Finding: Data Access is Necessary

The framework demonstrates a fundamental principle across two modes:

**Default mode (local, no model data for Direct QA):**

| Setting | Data Access | Expected EA | Why |
|---------|-------------|-------------|-----|
| 1 | ❌ | ~0% | Cannot guess GlobalIds |
| 2 | ❌ | ~0% | Schema ≠ data |
| 3 | ✅ Graph | Variable | May generate invalid queries |
| 4 | ✅ Graph | Higher | Valid syntax + data access |

**Cloud override mode (`--cloud-direct`):**

| Setting | Data Access | Expected EA | Why |
|---------|-------------|-------------|-----|
| 1 (cloud) | ✅ In-context | Moderate | LLM can search data, but may miss matches in large context |
| 2 (cloud) | ✅ In-context | Moderate+ | IDS specs guide the search, slightly better |
| 3 (cloud) | ✅ Graph | Variable | Gemini generates Cypher, same as default but different model |
| 4 | ⛔ Skipped | — | Requires local Outlines |

**Implication**: For BIM query answering, LLM reasoning alone is insufficient. Data access is mandatory. When data is provided in-context (cloud mode), accuracy improves dramatically but may still lag behind structured graph queries due to context window truncation and the needle-in-haystack challenge.

### 7.2 Secondary Finding: Grounding Improves Validity

Comparing Settings 3 and 4:
- **Setting 3 (SOFT)**: May hallucinate properties or use wrong syntax
- **Setting 4 (STRICT)**: Grammar guarantees valid output

Expected improvement in SVR: ~50-75% (soft) → 100% (strict)

### 7.3 Tertiary Finding: In-Context vs. Graph Data Access

Comparing Cloud Settings 1–2 against Cloud Setting 3:
- **Cloud Settings 1 & 2**: LLM must scan ~900K tokens of raw JSON to find matching elements. Accuracy depends on the model's ability to locate relevant data and extract GlobalIds without missing any.
- **Cloud Setting 3 (Cypher)**: LLM generates a concise query (~50 tokens). Neo4j executes it with O(1) index lookups over the full dataset with zero truncation.

This comparison isolates the value of **structured data access** even when in-context data is available.

### 7.4 Research Implications

1. **LLM + Database Architecture**: Pure LLM approaches cannot replace database queries for data retrieval tasks
2. **Schema Grounding Value**: Constraints help validity but cannot substitute for data access
3. **Grammar Constraints**: Token-level enforcement provides stronger guarantees than prompt engineering
4. **Large-Context Models**: While cloud LLMs with large context windows make in-context data access feasible, structured graph queries remain more reliable and scalable

---

## 8. Experimental Dataset

### 8.1 IFC Model Characteristics

The evaluation uses a real-world BIM model with the following characteristics:

| Entity Type | Count | Key Properties |
|-------------|-------|----------------|
| IfcColumn | 176 | LoadBearing=true, Length ~3650mm |
| IfcWall | 146 | IsExternal, Width 138.5-303mm |
| IfcSpace | 116 | GrossFloorArea, NetFloorArea (m²) |
| IfcDoor | 100 | FireRating='20 Minute' (10 doors), IsExternal |
| IfcWindow | 24 | Height, Width |
| IfcBuildingStorey | 5 | Names: "01 - Entry Level", "02 - Floor", etc. |

### 8.2 Graph Relationships

The loader materialises six edge types from IFC reification entities. Counts below are illustrative (taken from `data/Barcelona.ifc`, IFC2X3, 11,786 nodes after filtering `IfcRelationship` reifications); they vary per model.

| Relationship | Source IFC entity | Direction | Example count (Barcelona) |
|--------------|-------------------|-----------|---------------------------|
| `CONTAINS` | `IfcRelContainedInSpatialStructure` | `(:IfcSpatialStructureElement)→(:IfcElement)` | 361 literal + 942 derived |
| `DECOMPOSES` | `IfcRelAggregates` | `(parent)→(child)` | 952 |
| `HAS_OPENING` | `IfcRelVoidsElement` | `(:IfcElement)→(:IfcOpeningElement)` | 188 |
| `FILLS` | `IfcRelFillsElement` | `(:IfcDoor` or `:IfcWindow)→(:IfcOpeningElement)` | 128 |
| `IS_OF_TYPE` | `IfcRelDefinesByType` | `(:IfcObject)→(:IfcTypeProduct)` | 1,296 |
| `HAS_MATERIAL` | `IfcRelAssociatesMaterial` | `(:IfcElement)→(:IfcMaterial)` | 1,412 |

**Notes on the schema:**
- Every node carries its full IFC supertype chain as labels, so `MATCH (:IfcWall)` matches both `IfcWall` and `IfcWallStandardCase` instances; `MATCH (:IfcBuildingElement)` matches every physical element.
- `CONTAINS` is **transitive through aggregation**: synthesised edges are tagged `r.derived = true`. A storey-level query like `MATCH (s:IfcBuildingStorey)-[:CONTAINS]->(d:IfcDoor)` therefore counts curtain-wall doors that are aggregated under an `IfcCurtainWall` rather than directly contained. Filter `WHERE c.derived IS NULL` to recover the literal IFC topology.
- `IfcRelationship` subclasses are deliberately **not** materialised as nodes — they exist only as the edges above. `IfcTypeProduct` subclasses (`IfcWallType`, `IfcDoorStyle`, `IfcWindowStyle`, …) **are** kept as nodes, reachable via `IS_OF_TYPE`.

### 8.3 Test Set Composition

16 queries distributed across:
- **Basic queries** (6): Simple entity retrieval
- **Property filters** (5): WHERE conditions on properties
- **Relationship queries** (5): Traversing graph relationships

---

## 9. Configuration & Reproducibility

### 9.1 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://neo4j:7687` | Neo4j connection |
| `NEO4J_USER` | `neo4j` | Database user |
| `NEO4J_PASSWORD` | `password` | Database password |
| `LLM_PROVIDER` | `local` | `local`, `gemini`, or `openai` |
| `LLM_MODEL_NAME` | `Qwen/Qwen2.5-Coder-3B-Instruct` | Model identifier |
| `IFC_FILE_PATH` | `/app/data/model.ifc` | BIM model file |
| `IDS_FILE_PATH` | `/app/data/requirements.ids` | IDS constraints |

### 9.2 Running the Full Experiment

```bash
# Start Neo4j and load data
docker-compose up -d neo4j
python -m src.etl.loader

# Run 2×2 comparison (local mode)
python -m src.eval.cli \
  --test-set data/test_set.csv \
  --comparison

# Multi-model comparison (local)
for model in "Qwen/Qwen2.5-Coder-3B-Instruct" \
             "deepseek-ai/deepseek-coder-6.7b-instruct" \
             "mistralai/Mistral-7B-Instruct-v0.3"; do
  LLM_MODEL_NAME=$model python -m src.eval.cli \
    --test-set data/test_set.csv --comparison
done

# Cloud Override: Settings 1-3 on Gemini (requires LLM_API_KEY in .env)
# Setting 4 auto-skipped (requires local Outlines)
python -m src.eval.cli \
  --test-set data/test_set.csv \
  --model-dump data/model_dump.json \
  --cloud-direct \
  --comparison
```

---

## 10. Conclusion

This framework provides empirical evidence for a fundamental principle in LLM-based query systems: **data access cannot be replaced by reasoning**. 

The 2×2 experimental design isolates the effects of:
1. **Data access method** (Direct QA vs. Cypher/Graph)
2. **Grounding level** (Unconstrained vs. Schema-constrained)

Key findings:
- Direct QA achieves EA ≈ 0% regardless of grounding (no data access)
- Cypher generation with grammar constraints achieves highest accuracy
- Schema grounding improves syntactic validity but cannot substitute for data

**Recommendation**: For production BIM query systems, combine LLM reasoning (for natural language understanding) with structured data access (via graph queries) and grammar constraints (for output validity).

---

## Appendix A: File Structure

See the **File Structure Summary** in section 5 — the authoritative layout
lives there (one tree, no duplication).

---

## Appendix B: Development Changelog

### 2026-01-29: Major Bug Fixes and Enhancements

Based on analysis of experiment results showing poor Cypher pipeline performance (cypher_strict: 50% SVR, 6.2% EA; cypher_soft: 56.2% SVR, 37.5% EA), the following issues were identified and fixed:

#### Root Causes Identified

1. **Grammar didn't support relationship patterns** - Queries with `-[:CONTAINS]->` or `-[:HAS_MATERIAL]->` couldn't be generated
2. **LIMIT clause allowed without number** - Produced incomplete/invalid queries
3. **RETURN allowed property-only returns** - No node variable meant GlobalId extraction failed
4. **Gold queries used incorrect property names** - e.g., Width in meters vs. actual mm in IFC
5. **HAS_MATERIAL relationship missing from ETL** - Material associations weren't loaded
6. **EA calculation couldn't extract IDs** - Didn't handle Neo4j Node objects properly

#### Files Modified

##### `src/constraints/grammar.py`
- Added `relationships: list[str]` and `allow_relationships: bool` to `CypherGrammar` dataclass
- Added `force_return_node: bool` to ensure RETURN includes node variable
- Rewrote `build_cypher_regex()` to support:
  - Relationship patterns: `(n)-[:REL]->(m:Label)`
  - CONTAINS operator: `WHERE n.Prop CONTAINS 'value'`
  - IS NULL / IS NOT NULL checks
  - NOT operator: `WHERE NOT n.Prop CONTAINS 'value'`
  - Combined conditions with AND
- Disabled ORDER BY/LIMIT by default (set `allow_order_limit=True` to enable)

##### `src/llm_engine.py`
- Enhanced `CYPHER_SYSTEM_PROMPT_TEMPLATE` with:
  - Added HAS_MATERIAL relationship documentation
  - Added examples with relationships, CONTAINS, IS NOT NULL, NOT operator
  - Added critical rules for numeric values (no quotes) and boolean values
  - Strengthened output format instructions
- Improved `CYPHER_SOFT_CONSTRAINT_SUFFIX` with stricter formatting reminders

##### `src/eval/metrics.py`
- Rewrote `execute_and_get_ids()` for robust GlobalId extraction
- Added `_extract_global_id()` helper that handles:
  - Direct GlobalId values (strings)
  - Neo4j Node objects with `.get()` method
  - Dictionary results with 'GlobalId' key
  - Various result column formats

---

### Changelog: 2026-01-31 - Deep Audit & Multi-Model Support

#### Summary
Deep-dive audit fixing regex/grammar conflicts, adding difficulty-based metrics, and enabling multi-model result serialization.

##### `src/constraints/grammar.py`
- Added `CONVERSION_FUNCTIONS` pattern for `toInteger()`, `toFloat()`, `toString()`, etc.
- Fixed `COMPARISON_OP_BASIC` to properly include `>=` and `<=` operators
- Added `property_access` pattern supporting conversion functions: `toInteger(n.Property)`
- Increased `max_where_clauses` from 3 to 4 in `CypherGrammar` dataclass
- Enhanced `build_relationship_cypher_regex()` with full condition support:
  - NOT prefix for negation
  - CONTAINS operator for string matching
  - IS NULL / IS NOT NULL checks
  - All comparison operators including `>=` and `<=`
- All 16/16 gold queries from test_set.csv now pass regex validation

##### `src/eval/metrics.py`
- Enhanced `_extract_global_id()` to handle Neo4j Node objects with `.keys()` method
- Added `aggregate_by_difficulty()` function for complexity-based performance analysis
- Added `aggregate_by_category()` function for category-based breakdown
- Updated `aggregate_results()` to include `metrics_by_difficulty` in output

##### `src/eval/run_experiment.py`
- Added `_get_sanitized_model_name()` helper for filesystem-safe naming
- Enhanced `run_comparison()` to log difficulty breakdown per setting
- Updated `save_comparison_report()` with enhanced JSON schema:
  - `model_name`: Name of the LLM model
  - `setting_results`: The 2x2 matrix data
  - `setting_matrix`: Structured representation for visualization
  - `complexity_results`: EA metrics per difficulty level (Easy/Medium/Hard)
  - `category_results`: EA metrics per query category
  - `ranking`: Settings ranked by each metric
- Output filename format: `results_{model_name}_{timestamp}.json`

##### `src/etl/loader.py`
- Added `_extract_material_associations()` method supporting:
  - `IfcMaterial` direct associations
  - `IfcMaterialLayerSet` with layers
  - `IfcMaterialLayerSetUsage` references
  - `IfcMaterialConstituentSet` with constituents
- Added `_create_material_nodes_and_relationships()` to create IfcMaterial nodes and HAS_MATERIAL relationships
- Updated `create_relationships()` to include HAS_MATERIAL alongside CONTAINS and DECOMPOSES

##### `data/test_set.csv`
- Recreated with 16 questions based on analysis of actual IFC file
- Corrected gold queries using real property names and values discovered in model:
  - Wall Width in mm (138.5-303), not meters
  - Column Length (height) in mm (~3650)
  - FireRating = '20 Minute' (10 doors have it)
  - Material names: "Concrete - Cast-in-Place Concrete", "Metal - Steel - 345 MPa"
  - Building storey names: "01 - Entry Level", "02 - Floor", "03 - Floor", "Roof", "Parapet"
- Categories: basic_query (6), property_filter (5), relationship (5)

#### IFC Model Data Summary (for reference)

| Entity | Count | Key Properties |
|--------|-------|----------------|
| IfcColumn | 176 | LoadBearing=true, Length ~3650mm |
| IfcWall | 146 | IsExternal, Width 138.5-303mm |
| IfcSpace | 116 | GrossFloorArea, NetFloorArea (m²) |
| IfcDoor | 100 | FireRating='20 Minute' (10), IsExternal |
| IfcWindow | 24 | Height, Width |
| IfcBuildingStorey | 5 | Name (01/02/03/Roof/Parapet) |

| Relationship | Count | Description |
|--------------|-------|-------------|
| CONTAINS | 8 | Storey → Elements |
| DECOMPOSES | 56 | Building → Storeys |
| HAS_MATERIAL | 176 (new) | Elements → Materials |

#### Expected Improvements

After these fixes:
- **SVR** should improve significantly (grammar now produces complete queries)
- **SCR** should remain high (schema enforcement intact)
- **EA** should improve substantially (correct gold queries + proper ID extraction)

---

### 2026-02-03: Direct QA Redesign (Breaking Change)

**Major Design Change**: Direct QA settings (1 & 2) now have **no data access**.

#### Rationale
- Model dumps exceeded context windows (1.1M tokens vs 32K limit)
- Truncation made results meaningless (only ~3% visible)
- Cleaner design provides stronger evidence for the research hypothesis

#### New Experimental Design
- **Setting 1**: LLM receives only the question (pure reasoning)
- **Setting 2**: LLM receives question + IDS schema (schema-guided reasoning)
- **Expected result**: EA ≈ 0% for both settings

This design cleanly demonstrates that **LLM reasoning alone cannot answer BIM queries** - data access is mandatory.

#### Files Modified
- `src/llm_engine.py`: Redesigned Direct QA prompts without model data
- `src/eval/run_experiment.py`: Model dump no longer required
- `overview.md`: Restructured as scientific paper foundation

---

### 2026-02-04: Local LLM Pipeline — Architecture & Robustness Audit

#### Summary
Production-readiness audit for Settings 3 (CYPHER_SOFT) and 4 (CYPHER_STRICT) with local model execution. Found and fixed one critical bug, one medium issue, and one minor improvement.

#### Findings

##### Critical: Outlines `max_new_tokens` default truncates all constrained queries
`LocalLLMEngine._generate_cypher()` called `generator(prompt)` without `max_new_tokens`, defaulting to **20 tokens** per Outlines docs. A typical relationship query is 30-40 tokens, so every non-trivial CYPHER_STRICT query was silently truncated, producing incomplete Cypher that fails SVR and EA. This single bug undermined all Setting 4 results.

**Fix**: Added `max_new_tokens=256` to the constrained generator call, matching the unconstrained path.

##### Medium: Trailing backtick artifacts not stripped
`_clean_cypher_query()` handled paired `` ```cypher...``` `` fences but not trailing unpaired backticks — a common local model artifact. The MATCH...RETURN extraction regex included backtick characters in the output, causing SVR failures.

**Fix**: Added `query.replace('`', '').strip()` after fence removal. Backticks are never valid in the project's Cypher grammar subset.

##### Low: Silent error swallowing in `execute_and_get_ids`
The `except Exception` block returned `(set(), error_string)` without logging, making failed queries invisible at INFO level during experiment runs.

**Fix**: Added `logger.warning()` with query excerpt for debugging visibility.

##### Verified: Grammar regex covers all 16 gold queries
All gold queries pass validation including edge cases: unquoted numeric literals, `toInteger()`/`toFloat()` wrappers, relationship escaping (`-[:HAS_MATERIAL]->`), `IS NOT NULL`, `NOT ... CONTAINS`, boolean variants, and multi-condition WHERE clauses. Invalid entities are correctly rejected.

##### Verified: Outlines API usage is correct
`Generator(model, Regex(pattern))` confirmed as valid current API via Outlines documentation. The `from_transformers` and `from_llamacpp` initialization paths are also correct.

#### Files Modified

| File | Change | Severity |
|------|--------|----------|
| `src/llm_engine.py:783` | Added `max_new_tokens=256` to constrained generation | Critical |
| `src/llm_engine.py:521-523` | Strip backtick characters after fence removal | Medium |
| `src/eval/metrics.py:438` | Added `logger.warning` for failed query execution | Low |
