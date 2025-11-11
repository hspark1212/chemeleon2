# 1. Assets for Benchmark

## 1.1. PatchedPhaseDiagram for all entries with uncorrected energy

### MP-all

- `ppd-mp_all_entries_uncorrected_250409.pkl.gz` is a PatchedPhaseDiagram object created from all entries in the Materials Project database with uncorrected energies. The object is saved in a gzipped pickle format.

``` python
import os
import gzip
import pickle
import dotenv

from pymatgen.ext.matproj import MPRester
from pymatgen.analysis.phase_diagram import PatchedPhaseDiagram, PDEntry

dotenv.load_dotenv()
mp_api_key = os.getenv("MP_API_KEY")

# Collect all entries from the Materials Project database
all_entries = MPRester(mp_api_key).get_entries("", compatible_only=True)
print(f"Found {len(all_entries)} entries in the Materials Project database")

all_entries = [e for e in all_entries if e.data["run_type"] in ["GGA", "GGA_U"]]
print(f"Found {len(all_entries)} entries with GGA or GGA_U run type")

# Save energy with uncorrected energies
all_entries_uncorrected = [
    PDEntry(composition=e.composition, energy=e.uncorrected_energy) for e in all_entries
]
print(f"Found {len(all_entries_uncorrected)} entries with uncorrected energies")


# Create PatchedPhaseDiagram
ppd_mp = PatchedPhaseDiagram(all_entries_uncorrected, verbose=True)  # type: ignore
print(f"PatchedPhaseDiagram created with {len(ppd_mp.all_entries)} entries")

# Save the PatchedPhaseDiagram object
with gzip.open(
    "assets/ppd-mp_all_entries_uncorrected_250409.pkl.gz", "wb"
) as f:
    pickle.dump(ppd_mp, f)
print(
    "PatchedPhaseDiagram object saved as assets/ppd-mp_all_entries_uncorrected_250409.pkl.gz"
)
```

### Alex-MP-20

```python
from mattergen.evaluation.reference.reference_dataset_serializer import LMDBGZSerializer
from pymatgen.analysis.phase_diagram import PatchedPhaseDiagram, PDEntry

reference = LMDBGZSerializer().deserialize("data-release/alex-mp/reference_MP2020correction.gz")

# Save energy with uncorrected energies
all_entries_uncorrected = [
    PDEntry(composition=entry.composition, energy=entry.uncorrected_energy)
    for entry in reference
]

# Create PatchedPhaseDiagram
ppd_alex_mp_20 = PatchedPhaseDiagram(all_entries_uncorrected, verbose=True)

# Save the PatchedPhaseDiagram object
with gzip.open(
    "assets/ppd-alex_mp_20_entries_uncorrected_250730.pkl.gz", "wb"
) as f:
    pickle.dump(ppd_alex_mp_20, f)
print(
    "PatchedPhaseDiagram object for Alex-MP-20 saved as assets/ppd-alex_mp_20_entries_uncorrected_250730.pkl.gz"
)
```

### usage

``` python
import gzip
import pickle

from pymatgen.analysis.phase_diagram import PatchedPhaseDiagram
from pymatgen.analysis.phase_diagram import PDEntry

with gzip.open(
    "assets/ppd-mp_all_entries_uncorrected_250409.pkl.gz", "rb"
) as f:
    ppd_mp = pickle.load(f)
new_entry = PDEntry(composition="Li2O", energy=-2.5)  # Example entry
e_above_hull = ppd_mp.get_e_above_hull(new_entry, allow_negative=True)
```

## 1.2. All unique structures from the Materials Project database

- `mp_all_unique_structure_250416.json.gz` is a list of unique structures from the Materials Project database. The structures are saved in a gzipped JSON format. (148854 unique structures)

```python
import os
import dotenv

from monty.serialization import loadfn, dumpfn
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.ext.matproj import MPRester

dotenv.load_dotenv()
mp_api_key = os.getenv("MP_API_KEY")
assert mp_api_key is not None


with MPRester(mp_api_key) as mpr:
    docs = mpr.materials.search(material_ids={}, fields=["structure"])


st_list = [doc.structure for doc in docs]
print(f"Found {len(st_list)} structures")

# Unique structures
sm = StructureMatcher()
output = sm.group_structures(st_list)
unique_st_list = [o[0] for o in output]
print(f"Found {len(unique_st_list)} unique structures")


# save
dumpfn(unique_st_list, "assets/mp_all_unique_structure_250416.json.gz")
```

## 1.3. Random sampled 1000 structures in mp-20 test set for CSP task

- `csp_test_sampled_1000_compositions.txt` is a list of 1000 random sampled compositions from mp-20 test set. The compositions are saved in a text format for evaluation. (1000 unique compositions)

```python
df_test = pd.read_csv("../data/mp-20/test.csv", index_col=0)
print(len(df_test))

# Remove duplicates compositions
df_test = df_test.drop_duplicates(subset=["composition"])
print(len(df_test))

# Change original composition to raw_composition
df_test = df_test.rename(columns={"composition": "raw_composition"})

# Select 1000 random samples
n_sample = 1000
df_test_sampled = df_test.sample(n_sample, random_state=0)
df_test_sampled = df_test_sampled.reset_index(drop=True)

# Get compositions for CSP
st_list = [Structure.from_str(s, fmt="cif") for s in df_test_sampled["cif"]]
df_test_sampled["num_atoms"] = [len(st) for st in st_list]
df_test_sampled["composition"] = [
    str(st.composition).replace(" ", "") for st in st_list
]
print(len(df_test_sampled))

# Save to txt
with open("assets/csp_test_sampled_1000_compositions.txt", "w") as f:
    for c in df_test_sampled["composition"]:
        f.write(c + "\n")
```
