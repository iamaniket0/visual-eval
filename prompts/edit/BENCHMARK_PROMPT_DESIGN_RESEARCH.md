# Image Editing Benchmark Prompt Design: Deep Research Report

## Executive Summary

This report analyzes HOW state-of-the-art image editing benchmarks design their prompts — the taxonomies, difficulty calibration, linguistic structures, and atom decomposition methods used. Evidence is drawn from 9 benchmarks published 2023–2026.

---

## 1. Complex-Edit (UCSC, Apr 2025)

**The gold standard for complexity-controllable prompt design.**

### Methodology: Chain-of-Edit Pipeline

Complex-Edit introduces a 3-stage pipeline for generating prompts at precisely controlled complexity levels:

**Stage 1 — Sequence Generation:** GPT-4o generates a sequence of atomic instructions for each image, where each instruction maps to one of **24 predefined atomic operations** grouped into **9 categories**:

| Category | Atomic Operations |
|----------|-------------------|
| Object Manipulation & Transformation | Add, Remove, Replace, Move, Resize, Rotate, Duplicate an Object |
| Color & Tone Adjustments | Change Color, Apply Filter/Weather |
| Texture & Material Adjustments | Change Texture |
| Background & Environment | Change Background |
| Lighting & Shadows | Adjust Lighting |
| Text & Symbols | Add Text, Remove Text, Change Text Properties |
| Composition & Cropping | Crop Image, Zoom In/Out, Reframe Composition |
| Pose & Expression | Change Pose, Change Facial Expression |
| Special Effects | Add/Remove Particles, Add/Remove Special Effects |

**Stage 2 — Simplification:** Each atomic instruction is trimmed by GPT-4o to retain only core editing intent, removing commentary or extraneous details.

**Stage 3 — Instruction Compounding:** Simplified atomics are merged into compound instructions at varying complexity levels. A compound instruction at level C_i combines the first i atomic instructions.

### Complexity Levels (Exact Examples)

| Level | Example Prompt |
|-------|---------------|
| C1 | "Replace the dog with a cat." |
| C2 | "Replace the dog with a cat. Make the white vehicle blue." |
| C3 | "Replace the dog with a cat, remove the bicycle, and change the white vehicle's color to blue." |
| C8 | "Replace the dog with a cat, make the white vehicle blue, remove the bicycle, and then set the background to fall when raining, change the setting to an autumn night when raining, add a street light in the background and two sparrows resting on the fence." |

### Key Design Decisions
- Temperature increased from 1.0 → 1.15 during sequence generation for diversity
- CoT reasoning enabled in Stages 1 and 3
- Rule-based filtration catches garbled text at each stage
- GPT-4o integrates (not concatenates) instructions — e.g., "add yarn" + "change yarn color to red" → "add a red ball of yarn"

### Quantitative Findings
- Performance drops sharply from C1→C8 across all models
- At C8, even proprietary models (Imagen3) score ~4/10 on instruction following
- Open-source models (UltraEdit, AnyEdit) drop to near 0 at C5+
- Identity Preservation degrades most with complexity

---

## 2. GEditBench v2 (StepFun/NTU, Jun 2026)

**1,200 real-world queries across 23 tasks with open-set category.**

### Task Taxonomy (4 categories + open-set)

**Local Editing (10 tasks):**
- Subject Addition, Subject Removal, Subject Replace
- Size Adjustment, Color Alteration, Material Modification
- Portrait Beautification, Motion Change, Relation Change
- Text Editing

**Global Editing (5 tasks):**
- Background Change, Style Transfer, Tone Transfer
- Camera Motion, Line2Image, Enhancement (9 low-level subtasks)

**Reference Editing (3 tasks):**
- Character Reference, Object Reference, Style Reference

**Hybrid Editing:**
- Combines 3–5 basic edits into a single complex instruction

**Open-Set Editing (100 instances):**
- Trending real-world instructions that resist categorization
- Collected from Reddit and X (Twitter)

### Prompt Collection Methodology
- Real-world user editing instances collected from the Internet (Reddit, X)
- Trained experts manually filter instructions with similar intent
- Original user-uploaded images replaced with public images for privacy
- Instructions written in natural language as users would express them

### Difficulty Calibration
- **Hybrid** category explicitly tests complex multi-operation instructions
- **Open-set** category covers out-of-distribution instructions
- No explicit easy/medium/hard labels, but difficulty emerges from task type (e.g., Relation Change and Chart Editing are inherently harder)

### Novel Tasks Introduced
- In-Image Text Translation (multilingual poster production)
- Chart Editing (chart refinement and type transformation)
- Enhancement (blur, compression, moiré, low-light, noise, flare, reflection, haze, rain restoration)

---

## 3. ImgEdit-Bench (PKU, May 2025 — NeurIPS 2025)

**Three-tier difficulty with explicit UGE (Understanding-Grounding-Editing) suite.**

### Edit Type Taxonomy (10 single-turn + 3 multi-turn)

**Single-Turn:**
| Category | Types |
|----------|-------|
| Local Edit | Add, Remove, Replace, Alter, Motion Change, Object Extraction |
| Global Edit | Background Replacement, Style/Tone Transfer |
| Visual Edit | Edit using a reference image with identity consistency |
| Hybrid Edit | Two local operations combined (e.g., "add a scarf and change the cat's fur to white") |

**Multi-Turn (3 types):**
- **Content Memory:** Global constraint stated once must persist (e.g., "all generation must have wooden texture")
- **Content Understanding:** Pronouns/references resolve to earlier turns (e.g., "Make it black" referring to previously added clothing)
- **Version Backtracking:** "Undo the previous change" or edit based on earlier versions

### Three Difficulty Tiers

1. **Basic-Edit Suite (734 test cases):**
   - Ten representative concepts from 6 super-categories (human, transportation, nature, animals, architecture, necessities)
   - 5 prompts per concept for add tasks
   - Prompts range from short to elaborate
   - Instructions generated by GPT-4o, then manually filtered

2. **Understanding-Grounding-Editing (UGE) Suite (47 images):**
   - Manually curated complex images with: partially occluded targets, multiple same-category instances, camouflaged objects, uncommon editing subjects
   - Prompts require: spatial reasoning, multi-object coordination, compound operations, large-scale modifications

3. **Multi-Turn Suite (30 cases):**
   - 10 images × 3 tasks, 3 interaction turns each
   - Tests content memory, understanding, and version backtracking

### Prompt Design Principles
- GPT-4o generates prompts using caption + edit type + bounding box + target object as conditioning
- Position of object and approximate size embedded in instruction using bounding box
- Multi-turn prompts: few-shot examples → model produces entire dialogue → split into turns
- Each dialogue limited to 2–3 turns from 4 basic operations (add, remove, replace, alter)

### Quantitative Metrics
- 8.7k unique words in prompt vocabulary (vs. 2k for MagicBrush, 3.7k for UltraEdit)
- Average resolution: 1280px (short side)
- Hybrid edits explicitly combine operations from different types

---

## 4. ICE-Bench (Alibaba/Tongyi Lab, Mar 2025 — ICCV 2025)

**31 fine-grained tasks with coarse-to-fine hierarchy.**

### Hierarchical Task Organization

**Level 1 — Generation Type:** Creating vs. Editing
**Level 2 — Reference Dependency:** No-Ref vs. Ref
**Level 3 — 31 Fine-Grained Tasks**

The 4 coarse categories:
1. **No-Ref Image Creating** (text-to-image)
2. **Ref Image Creating** (reference-guided generation)
3. **No-Ref Image Editing** (instruction-based editing)
4. **Ref Image Editing** (reference + instruction)

### Instruction Design
Instructions use a structured format with placeholders:

| Task | Example Instruction |
|------|-------------------|
| Depth-guided creating | "Craft an artistic work using depth map in [SOURCE] to match given description" |
| Subject reference | "Refer to the subject in [REF_1], A stuffed bunny is holding an apple and standing on the table." |
| Text overlay | "Overlay the 'MAKE' on the mask portion of the [SOURCE]." |
| Subject-guided editing | "Alter the mask in [SOURCE] based on the subject in [REF_1]." |
| Expression editing | "Make the girl in the [SOURCE] smile." |

### Data Construction
- 6,538 task instances
- Hybrid data: real scenes + virtual generation
- VLLM-QA metric: uses large models to assess editing success via question-answering

### Complexity Dimension
- No explicit difficulty levels
- Complexity emerges from task type (single-input vs. multi-input, local vs. global)
- Does NOT include "instruction complexity" as a separate dimension (despite our initial hypothesis)

---

## 5. EditVal (Arizona State/USC, Oct 2023)

**Atom-based modular evaluation with 13 edit types.**

### Edit Type Suite (13 types)

1. Object Addition
2. Object Replacement
3. Positional Addition
4. Size
5. Position Replacement
6. Alter Parts
7. Background
8. Texture
9. Style
10. Color
11. Shape
12. Action
13. Viewpoint

### Atom Decomposition Methodology

EditVal treats each edit as a single **atomic operation** applied to a specific object class. The benchmark is structured as:

```json
{
  "bench": {
    "240698": {
      "object-addition": [
        {"to": ["bag", "cup", "ball", "books", "shoes"]}
      ],
      "color": [
        {"from": "brown", "to": ["red", "blue", "green", "yellow"]}
      ],
      "texture": [
        {"from": "wooden", "to": ["metal", "stone", "glass"]}
      ]
    }
  }
}
```

### Prompt Generation Process
1. ChatGPT identifies which MS-COCO classes are plausible for each edit type
2. For each (class, edit-type) pair, ChatGPT generates specific operations (e.g., "What objects can be added to a Bench?" → ["ball", "cup", "books"])
3. Prompts constructed from templates: "Add a cup to the bench", "Change the bench color from brown to red"

### Evaluation Scoring
- Human study: 3-point scale per question (0="not applied" to 3="perfectly applied")
- Three questions per edit: (1) edit accuracy, (2) object properties preserved, (3) image context preserved
- Automated: Binary score using OwL-ViT object detection (edit correct or not)
- 648 unique operations across 92 images in 19 MS-COCO classes

### Key Finding on Difficulty
- Non-spatial edits (object-addition, color, texture) → higher success
- Spatial edits (position-replacement, positional-addition) → near-zero accuracy across all methods
- No explicit difficulty labels, but difficulty is inherent in the edit type

---

## 6. KontextBench (Black Forest Labs, Jun 2025)

**1,026 crowd-sourced real-world image-prompt pairs across 5 task categories.**

### Task Categories and Distribution

| Category | Count | Description |
|----------|-------|-------------|
| Instruction Editing - Local | 416 | Targeted element modifications |
| Instruction Editing - Global | 262 | Broad scene-level changes |
| Character Reference | 193 | Preserving characters across scenes |
| Text Editing | 92 | Modifying text within images |
| Style Reference | 63 | Applying styles from reference images |

### Prompt Examples (from dataset)

**Local Editing:**
- "give the cat a tophat"
- "make the cat very fat"
- "make the cat orange"
- "remove the hand in the middle"
- "Add hats in the boxes" (visual cue editing)
- "The woman is now wearing a green dress, the painting in the back now shows a beach scene, the text on the TV says 'Kontext' now"

**Global Editing:**
- "Turn this into a neon sign hanging on a brick wall"
- "Change the background to purple"
- "she is now taking a selfie in the streets of Freiburg, it's a lovely day out."
- "it's now snowing, everything is covered in snow."

**Text Editing:**
- "Edit the word 'Lover' to 'Hater'"
- "Replace 'MONTREAL' with 'FREIBURG'"
- "Replace 'SYNC & BLOOM' with 'FLUX & JOY'"

**Style Reference:**
- "Using this style, a kid on a bicycle rolls through desert ruins"
- "Using this style, a dapper octopus conducts a jazz duo of owls on a shimmering moonlit bandstand."

**Character Reference:**
- "The bird is now sitting in a bar and enjoying a beer."
- "There are now two of these birds."
- "Watch them from behind."

### Prompt Design Philosophy
- **Crowd-sourced from real users** — not synthetically generated
- Derived from 108 base images (personal photos, CC-licensed art, public domain, AI-generated)
- Designed to capture real-world usage patterns
- No explicit difficulty levels — reflects natural distribution of user requests

### Multi-Turn Design
- Iterative application: each output serves as input for next edit
- Tests character consistency drift across successive edits

---

## 7. Soft-TIFA (Meta/GenEval 2, Dec 2025)

**Template-based atom decomposition with soft probabilistic scoring.**

### Atom Definition

In GenEval 2, an "atom" is a visual primitive within a compositional prompt:
- **Object atoms:** "bicycles", "cows", "sheep"
- **Attribute atoms:** "white", "plastic", "blue", "purple"
- **Relation atoms:** "in front of", "playing with", "behind"
- **Count atoms:** "four", "three", "seven"

### Atomicity (Compositionality Measure)

The atomicity of a prompt = number of distinct atoms. "a" and "and" don't count.

| Atomicity | Example Prompt | Atom Breakdown |
|-----------|---------------|----------------|
| 3 | "two metal toys" | count=two, attribute=metal, object=toys |
| 5 | "a blue sheep playing with three pigs" | attribute=blue, object=sheep, relation=playing with, count=three, object=pigs |
| 7 | "four white bicycles in front of three plastic cows" | count=four, attribute=white, object=bicycles, relation=in front of, count=three, attribute=plastic, object=cows |
| 10 | "seven purple flamingos playing with a green sheep behind five metal croissants" | 10 atoms |

### Prompt Template Structure

```
{count1 or "a"}{attribute1}{object1}{relation1 or "and"}{count2 or "a"}{attribute2}{object2}
```

Extended to 3 objects. Counts, attributes, and relations are optional → varying compositionality.

### Scoring Method

For each atom, a template question is generated (also from templates, not LLM):
- Object: "Is there a {object} in this image?"
- Attribute: "Is the {object} {attribute}?"
- Count: "Are there {count} {object}s?"
- Relation: "Is the {object1} {relation} the {object2}?"

**Soft-TIFA_AM** (Arithmetic Mean): Average P(Yes) across atoms → captures atom-level performance

**Soft-TIFA_GM** (Geometric Mean): Geometric mean of P(Yes) → captures prompt-level performance (penalizes any single failure)

### Quantitative Results
- GenEval 2: 800 prompts, 100 at each atomicity level (3–10)
- Best model (Gemini 2.5 Flash): 84.4% atom-level, only 31% prompt-level
- Performance drops sharply with compositionality: ~80% at atomicity 3, near 0% at atomicity 10
- AUROC of 94.5% with human alignment (vs. 92.4% VQAScore, 91.6% TIFA)

### Vocabulary
- 40 objects (20 animate, 20 inanimate; 20 from COCO, 20 otherwise)
- 18 attributes (colors, materials, patterns)
- 9 relations (3D spatial prepositions + transitive verbs)
- 6 counts (2–7)

---

## 8. REDEdit-Bench (FireRedTeam, Feb 2026)

**1,542 bilingual (Chinese-English) editing pairs across 15 categories.**

### 15 Task Categories with Counts

| Category | Count | Example Prompt |
|----------|-------|---------------|
| Add | 143 | "Add a seven-spotted ladybug on the green plant in the picture" |
| Adjust | 156 | (Attribute modification) |
| Background | 91 | (Background modification) |
| Beauty | 79 | (Beauty enhancement) |
| Color | 99 | (Color modification) |
| Compose | 100 | (Image composition) |
| Extract | 95 | (Element extraction) |
| Lowlevel | 47 | (Low-level processing: denoising, deblurring) |
| Motion | 78 | (Motion addition) |
| Portrait | 102 | (Portrait editing) |
| Remove | 147 | (Object removal) |
| Replace | 140 | (Object replacement) |
| Stylize | 92 | (Style transfer) |
| Text | 123 | (Text editing) |
| Viewpoint | 50 | (Viewpoint change) |

### Prompt Examples (from HuggingFace dataset)
- "Add a white heart-shaped latte art in the coffee cup"
- "Add a man running in sportswear on the road"
- "Add a seven-spotted ladybug on the green plant in the picture"

### Prompt Construction Methodology
- Images collected from >3,000 internet sources
- Expert curation by trained annotators
- Bilingual: each instruction has both Chinese and English versions
- Instructions designed to "align with human language usage"

### Evaluation Dimensions (not difficulty levels)
- **Alignment** — instruction following accuracy
- **Consistency** — visual coherence in edits
- **Realism** — naturalness of generated content
- **Aesthetics** — visual quality

### Difficulty Calibration
REDEdit-Bench does NOT implement explicit difficulty tiers. Instead:
- Difficulty is implicitly encoded in task type (e.g., "viewpoint" harder than "color")
- Expert curation ensures "diverse scenarios"
- The benchmark paper focuses more on model evaluation than prompt difficulty calibration

---

## 9. TIEdit (Shanghai Jiao Tong Univ., Mar 2026)

**512 source images × 8 tasks × 10 models = 5,120 edited images with 15,360 MOSs.**

### Eight Editing Tasks

| Task | Description | Scope |
|------|-------------|-------|
| Object Addition | Insert a new object into the scene | Local |
| Object Removal | Remove an object and restore background | Local |
| Object Replacement | Replace an object with another one | Local |
| Appearance Modification | Change object color, material, or texture | Local |
| Action Modification | Modify the pose or action of a subject | Local |
| Emotion Modification | Change facial expression or affective cues | Local |
| Re-contextualization | Alter the global environment or scene context | Global |
| Artistic Stylization | Transform the image into an artistic style | Global |

### Prompt Structure (Unified Triple Template)

Each prompt has three components:
1. **Source Description:** "A red car is driving on the highway."
2. **Editing Instruction:** "Change the red car to a blue truck."
3. **Target Description:** "A blue truck is driving on the highway."

This design supports both description-based models (using target description) and instruction-based models (using editing instruction).

### Prompt Examples
- "Add a boat to the lake."
- "Paint floral patterns on the mug."
- "Change the red butterfly to a blue butterfly."

### Complexity Calibration
- ~10% of prompts intentionally include **spatial relationships or compositional reasoning**
- Generated by GPT-4 with structured templates, then human-reviewed
- No explicit easy/medium/hard labels
- The benchmark is designed as a quality assessment tool (MOS-based) rather than difficulty-stratified

### Evaluation Scale
- 307,200 raw ratings from 20 experts over 5 months
- Five-point scale across 3 dimensions
- Editing alignment shows broadest distribution (greatest difficulty variance)
- Content preservation and perceptual quality cluster at higher scores

---

## 5. MagicBrush (OSU/Waterloo, NeurIPS 2023)

**10K+ manually annotated triplets for multi-turn editing.**

### Multi-Turn Instruction Design

**Session Structure:**
- Each "session" starts from one source image (MS-COCO)
- Max 3 edit turns per session
- Each turn: (source_image, instruction, target_image)
- Multi-turn: error accumulates across turns

**Distribution:**
- 5,313 total sessions
- 2,105 sessions with 1 edit
- 1,341 sessions with 2 edits
- 1,867 sessions with 3 edits

### Prompt Examples (from paper)

**Multi-turn session example:**
- Turn 1: "Let the cat have blue eyes"
- Turn 2: "Let it be angry and hiss"
- Turn 3: "Wear it a necklace"

**Another session:**
- Turn 1: "Have the woman be playing a guitar"
- Turn 2: "Add a barn in the background"
- Turn 3: "Add a bale of hay in field"

**Single-turn examples:**
- "Remove the wooden frame"
- "Put a smiley face on the yellow light"
- "Make background a county fair"
- "Have him a cowboy hat"
- "Change the shirt to plaid"
- "Add a stream by the sheep"

### Prompt Categories (from keyword analysis)
- Object addition/replacement/removal
- Action changes
- Color alterations
- Text or pattern modifications
- Object quantity adjustments

### Annotation Methodology
- Workers propose instructions themselves (not template-generated)
- Workers interact with DALL-E 2 using various prompts/masks until satisfied
- If no qualified target achievable after several trials → example dropped
- Post-verification: consistency and image quality manually checked

### Difficulty Characteristics
- No explicit difficulty labels
- Multi-turn is inherently harder (error accumulation)
- Mask-free multi-turn = hardest scenario
- Quality scores: consistency 4.1/5, image quality 3.9/5

---

## Cross-Benchmark Synthesis: Prompt Design Principles

### 1. Complexity Calibration Strategies

| Benchmark | Strategy | Levels |
|-----------|----------|--------|
| Complex-Edit | Number of compounded atomic operations | 8 levels (C1–C8) |
| GenEval 2/Soft-TIFA | Number of atoms (objects+attributes+relations) | 8 levels (3–10 atoms) |
| ImgEdit-Bench | Separate test suites (Basic → UGE → Multi-turn) | 3 tiers |
| GEditBench v2 | Task type hierarchy + Hybrid + Open-set | Implicit via task difficulty |
| Others (KontextBench, REDEdit, TIEdit, EditVal) | No explicit difficulty stratification | Task-type varies difficulty |

### 2. Prompt Generation Methods

| Method | Used By | Pros | Cons |
|--------|---------|------|------|
| GPT-4/4o from structured templates | Complex-Edit, ImgEdit, TIEdit | Scalable, diverse | May introduce bias |
| Crowd-sourced from real users | KontextBench, GEditBench v2 | Authentic | Limited scale |
| Manual annotation by workers | MagicBrush, REDEdit-Bench | High quality | Expensive |
| ChatGPT + MS-COCO classes | EditVal | Systematic | Limited to COCO vocabulary |

### 3. Linguistic Structure Patterns

| Pattern | Example | Used By |
|---------|---------|---------|
| Imperative command | "Remove the bicycle" | All benchmarks |
| Descriptive target | "The cat is now wearing a hat" | KontextBench |
| Conditional/contextual | "she is now taking a selfie in Freiburg" | KontextBench |
| Multi-operation compound | "add X, change Y, and remove Z" | Complex-Edit, ImgEdit (Hybrid) |
| Pronoun reference (multi-turn) | "Make it black" | ImgEdit multi-turn |
| Template with placeholders | "[SOURCE]", "[REF_1]" | ICE-Bench |
| Source→Target description | "red car" → "blue truck" | TIEdit |

### 4. What Makes Prompts Non-Trivial

| Difficulty Factor | Evidence |
|-------------------|----------|
| Multiple operations in one instruction | Complex-Edit shows sharp degradation C1→C8 |
| Spatial reasoning requirements | EditVal shows 0–15% accuracy for position edits |
| Pronoun/reference resolution | ImgEdit multi-turn tests this explicitly |
| Compositional attribute binding | GenEval 2 shows attribution errors multiply with atoms |
| Uncommon/out-of-distribution edits | GEditBench v2 open-set category |
| Fine-grained identity preservation | All benchmarks measure this but few test it explicitly |
| Counting constraints | GenEval 2 shows counting remains hard (60% at best) |

### 5. Quantitative Prompt Statistics

| Benchmark | Prompts | Avg. Operations/Prompt | Vocabulary Size | Word Count Range |
|-----------|---------|----------------------|-----------------|-----------------|
| Complex-Edit | 1,600 (200 imgs × 8 levels) | 1–8 | N/A | Scales with level |
| GenEval 2 | 800 | 3–10 atoms | 73 concepts | 3–15 words |
| GEditBench v2 | 1,200 | 1–5 (Hybrid) | Real-world | Natural language |
| ImgEdit-Bench | 811 | 1–2 | 8.7k unique words | Short to elaborate |
| KontextBench | 1,026 | 1–3 | Natural | 3–30 words |
| EditVal | 648 | 1 (atomic only) | MS-COCO 19 classes | Template-based |
| MagicBrush | 10,388 turns | 1 per turn | 2k unique words | Short imperative |
| REDEdit-Bench | 1,542 | 1 | Bilingual | Natural language |
| TIEdit | 512 | 1 (~10% compositional) | GPT-4 generated | Structured triple |

---

## Key Takeaways for Prompt Design

1. **Atom-based complexity control is the most rigorous approach** — Complex-Edit and GenEval 2 demonstrate that defining atomic operations and composing them gives precise control over difficulty.

2. **Real-world prompts are linguistically different from synthetic ones** — KontextBench and GEditBench v2 show users write contextually ("she is now taking a selfie") rather than imperatively ("change background to street").

3. **Difficulty isn't just about word count** — Spatial reasoning, identity preservation, and compositional attribute binding are orthogonal difficulty axes that most benchmarks test implicitly but few control explicitly.

4. **Multi-turn adds a unique difficulty dimension** — Error accumulation, pronoun resolution, and version backtracking (ImgEdit, MagicBrush) test capabilities that single-turn prompts cannot.

5. **No benchmark fully crosses all dimensions** — Complex-Edit controls complexity but uses only synthetic generation for evaluation; KontextBench captures real usage but lacks difficulty stratification.
