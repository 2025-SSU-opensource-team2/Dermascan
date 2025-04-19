import os

def extract_fine_classes(test_dir, log_path='log.txt'):
    fine_labels = set()

    # Traverse all class folders
    for coarse_class in os.listdir(test_dir):
        coarse_path = os.path.join(test_dir, coarse_class)
        if not os.path.isdir(coarse_path):
            continue

        for fname in os.listdir(coarse_path):
            if fname.endswith(('.jpg', '.jpeg', '.png')):
                # Extract fine-grained class from filename (before last dash)
                name = os.path.splitext(fname)[0]  # remove extension
                parts = name.rsplit('-', 1)  # split from the right
                if len(parts) == 2:
                    fine_label = parts[0].strip().lower()
                    fine_labels.add(fine_label)

    fine_labels = sorted(list(fine_labels))

    # Write to log file
    with open(log_path, 'w') as f:
        f.write(f"Total fine-grained classes: {len(fine_labels)}\n\n")
        for label in fine_labels:
            f.write(f"{label}\n")

    print(f"✔ Fine-grained class info saved to '{log_path}'.")

from collections import defaultdict
import os

def cluster_fine_labels(log_path="log.txt", min_group=3, top_n=30):
    with open(log_path, 'r') as f:
        lines = f.readlines()[2:]  # skip header
        fine_labels = [line.strip() for line in lines]

    buckets = defaultdict(list)
    for label in fine_labels:
        prefix = label.split('-')[0]  # take first word as grouping key
        buckets[prefix].append(label)

    merged = {}
    for prefix, group in buckets.items():
        if len(group) >= min_group:
            for label in group:
                merged[label] = prefix  # merge into parent group

    # Optional: see most common groupings
    print(f"Top {top_n} grouped categories:")
    for prefix, group in sorted(buckets.items(), key=lambda x: -len(x[1]))[:top_n]:
        print(f"{prefix}: {len(group)}")

    # Save merged mapping
    with open("merged_mapping.txt", 'w') as f:
        for fine, coarse in merged.items():
            f.write(f"{fine} -> {coarse}\n")

    print(f"\n✔ Merged {len(merged)} fine labels into {len(set(merged.values()))} coarse categories.")


import os
from collections import defaultdict

FINE_TO_COARSE = {
    # === acne ===
    "acne-cystic": "acne",
    "acne-open-comedo": "acne",
    "acne-pustular": "acne",
    "acne-closed-comedo": "acne",
    "acne-keloidalis": "acne",

    # === actinic-keratosis ===
    "actinic-keratosis-lesion": "actinic-keratosis",
    "actinic-keratosis-5fu": "actinic-keratosis",
    "actinic-keratosis-face": "actinic-keratosis",
    "actinic-keratosis-horn": "actinic-keratosis",

    # === alopecia ===
    "alopecia-areata": "alopecia",

    # === basal-cell-carcinoma ===
    "basal-cell-carcinoma-face": "basal-cell-carcinoma",
    "basal-cell-carcinoma-lesion": "basal-cell-carcinoma",
    "basal-cell-carcinoma-nose": "basal-cell-carcinoma",
    "basal-cell-carcinoma-pigmented": "basal-cell-carcinoma",
    "basal-cell-carcinoma-superficial": "basal-cell-carcinoma",

    # === blistering-autoimmune ===
    "bullous-pemphigoid": "blistering-autoimmune",
    "pemphigus": "blistering-autoimmune",

    # === candidiasis ===
    "candidiasis-large-skin-folds": "candidiasis",

    # === cutaneous-lymphoma ===
    "ctcl": "cutaneous-lymphoma",

    # === dermatitis ===
    "perioral-dermatitis": "dermatitis",
    "rhus-dermatitis": "dermatitis",
    "stasis-dermatitis": "dermatitis",
    "allergic-contact-dermatitis": "dermatitis",
    "dermatitis-herpetiformis": "dermatitis",

    # === drug-induced ===
    "drug-eruptions": "drug-induced",
    "fixed-drug-eruption": "drug-induced",
    "phototoxic-reactions": "drug-induced",

    # === eczema ===
    "eczema-hand": "eczema",
    "eczema-nummular": "eczema",
    "eczema-chronic": "eczema",
    "eczema-fingertips": "eczema",
    "eczema-foot": "eczema",
    "eczema-herpeticum": "eczema",
    "eczema-subacute": "eczema",

    # === erythema ===
    "erythema-multiforme": "erythema",

    # === fibrous-tumors ===
    "dermatofibroma": "fibrous-tumors",
    "keloids": "fibrous-tumors",
    "epidermal-cyst": "fibrous-tumors",

    # === granulomatous-diseases ===
    "granuloma-annulare": "granulomatous-diseases",

    # === herpes ===
    "herpes-zoster": "herpes",
    "herpes-cutaneous": "herpes",
    "herpes-simplex": "herpes",
    "herpes-type-1-primary": "herpes",
    "herpes-type-1-recurrent": "herpes",
    "herpes-type-2-primary": "herpes",

    # === hidradenitis-suppurativa ===
    "hidradenitis-suppurativa": "hidradenitis-suppurativa",

    # === hypopigmentation ===
    "vitiligo": "hypopigmentation",

    # === infection ===
    "intertrigo": "infection",
    "scabies": "infection",
    "tick-bite": "infection",
    "varicella": "infection",
    "viral-exanthems": "infection",
    "erosio-interdigitalis-blastomycetica": "infection",

    # === lichen ===
    "lichen-planus": "lichen",
    "lichen-sclerosus-penis": "lichen",
    "lichen-sclerosus-skin": "lichen",
    "lichen-simplex-chronicus": "lichen",

    # === lipid-disorder ===
    "xanthomas": "lipid-disorder",

    # === lupus ===
    "lupus-chronic-cutaneous": "lupus",

    # === melanoma ===
    "malignant-melanoma": "melanoma",

    # === metabolic-disorder ===
    "porphyrias": "metabolic-disorder",

    # === molluscum-contagiosum ===
    "molluscum-contagiosum": "molluscum-contagiosum",

    # === nevus ===
    "nevus-sebaceous": "nevus",
    "congenital-nevus": "nevus",
    "epidermal-nevus": "nevus",
    "melanocytic-nevi": "nevus",
    "lentigo-adults": "nevus",
    "atypical-nevi": "nevus",
    "atypical-nevi-dermoscopy": "nevus",

    # === nodular-conditions ===
    "chondrodermatitis-nodularis": "nodular-conditions",

    # === nail-disorders ===
    "chronic-paronychia": "nail-disorders",
    "distal-subungual-onychomycosis": "nail-disorders",
    "onycholysis": "nail-disorders",

    # === genodermatoses ===
    "dariers-disease": "genodermatoses",
    "neurofibromatosis": "genodermatoses",

    # === urticaria ===
    "hives-urticaria-acute": "urticaria",

    # === pityriasis ===
    "pityriasis-rosea": "pityriasis",
    "pityriasis-rubra-pilaris": "pityriasis",

    # === psoriasis ===
    "psoriasis": "psoriasis",
    "psoriasis-chronic-plaque": "psoriasis",
    "psoriasis-guttate": "psoriasis",
    "psoriasis-palms-soles": "psoriasis",
    "psoriasis-pustular-generalized": "psoriasis",
    "psoriasis-scalp": "psoriasis",

    # === rosacea ===
    "rosacea": "rosacea",
    "rosacea-nose": "rosacea",

    # === sclerosing-conditions ===
    "morphea": "sclerosing-conditions",

    # === sebaceous ===
    "sebaceous-hyperplasia": "sebaceous",

    # === seborrheic-keratosis ===
    "seborrheic-dermatitis": "seborrheic-keratosis",
    "seborrheic-keratoses-ruff": "seborrheic-keratosis",
    "seborrheic-keratoses-smooth": "seborrheic-keratosis",
    "seborrheic-keratosis-irritated": "seborrheic-keratosis",
    "stucco-keratoses": "seborrheic-keratosis",

    # === squamous-cell-carcinoma ===
    "bowens-disease": "squamous-cell-carcinoma",
    "keratoacanthoma": "squamous-cell-carcinoma",

    # === tinea ===
    "tinea-body": "tinea",
    "tinea-face": "tinea",
    "tinea-foot-dorsum": "tinea",
    "tinea-foot-webs": "tinea",
    "tinea-groin": "tinea",
    "tinea-hand-dorsum": "tinea",
    "tinea-incognito": "tinea",
    "tinea-scalp": "tinea",
    "tinea-versicolor": "tinea",

    # === vascular-anomaly ===
    "hemangioma": "vascular-anomaly",
    "hemangioma-infancy": "vascular-anomaly",
    "pyogenic-granuloma": "vascular-anomaly",
    "cherry-angioma": "vascular-anomaly",
    "venous-lake": "vascular-anomaly",
    "venous-malformations": "vascular-anomaly",

    # === vascular-inflammation ===
    "vasculitis": "vascular-inflammation",
    "schamberg-disease": "vascular-inflammation",

    # === warts ===
    "warts": "warts",
    "warts-common": "warts",
    "warts-digitate": "warts",
    "warts-flat": "warts",
    "warts-plantar": "warts",
    "genital-warts": "warts"
}


DISCARD_FINE_LABELS = [
    "skin-tags-polyps",
    "neurotic-excoriations",
    "sun-damaged-skin",
    "crest-syndrome",
    "tuberous-sclerosis",
    "mucous-cyst",
    "pilar-cyst",
    "aids",
    "dermagraphism",
    "perleche"
]

UNNECESSARY_KEYWORDS = [
    'img', 'box', 'misc', 'habit', 'variation', 'forest', 'angry-back',
    'earimg', 'dryfeet', 'fire-ants', 'patch-testing', 'test',
    'bed-ridges', 'miscimg', 'scalpimg', 'unknown'
]

EXCEPTION_LABELS = {'05atopic031011'}

def is_discardable(label):
    if label in EXCEPTION_LABELS:
        return False
    return any(key in label for key in UNNECESSARY_KEYWORDS)

def get_coarse_label(fine_name):
    if fine_name in FINE_TO_COARSE:
        return FINE_TO_COARSE[fine_name]
    for keyword, coarse in FINE_TO_COARSE.items():
        if keyword in fine_name:
            return coarse
    return 'other'

def generate_refined_log(test_dir, log_path='log_grouped.txt', min_fine_samples=11):
    test_dir = os.path.expanduser(test_dir)

    fine_label_counts = defaultdict(int)
    fine_to_coarse = {}

    for coarse_class in os.listdir(test_dir):
        coarse_path = os.path.join(test_dir, coarse_class)
        if not os.path.isdir(coarse_path):
            continue

        for fname in os.listdir(coarse_path):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
        
            fine = os.path.splitext(fname)[0].rsplit('-', 1)[0].lower()

            if is_discardable(fine) or fine in DISCARD_FINE_LABELS:
                continue

            fine_label_counts[fine] += 1
            if fine not in fine_to_coarse:
                fine_to_coarse[fine] = get_coarse_label(fine)

    # Now filter valid and excluded fine labels
    valid_fine = {f for f, c in fine_label_counts.items() if c >= min_fine_samples}
    excluded_fine = {f for f, c in fine_label_counts.items() if c < min_fine_samples}

    # Group valid fine labels by coarse label
    grouped = defaultdict(list)
    for fine in valid_fine:
        grouped[fine_to_coarse[fine]].append(fine)

    with open(log_path, 'w') as f:
        f.write("[Valid Groups for Classification]\n\n")
        f.write(f"Total valid coarse groups: {len(grouped)}\n")
        f.write(f"Total valid fine labels: {len(valid_fine)}\n\n")  
        for coarse in sorted(grouped.keys()):
            fine_list = sorted(grouped[coarse])
            f.write(f"{coarse} ({len(fine_list)} fine labels)\n")
            for fine in fine_list:
                f.write(f"  - {fine} ({fine_label_counts[fine]})\n")
            f.write("\n")

        f.write("[Excluded Fine Labels (≤ 10 images)]\n\n")
        for fine in sorted(excluded_fine):
            f.write(f"{fine} ({fine_label_counts[fine]})\n")

    print(f"✔ Refined log saved to: {log_path}")

# Example usage
generate_refined_log(
    test_dir="~/Dermascan/dermnet_data/test",
    log_path="log_grouped.txt"
)



# Example usage
# cluster_fine_labels(log_path="dataCheck_log.txt")

# Example usage
# extract_fine_classes(test_dir="/home/dwlkb01/Dermascan/dermnet_data/test/", log_path="dataCheck_log.txt")
