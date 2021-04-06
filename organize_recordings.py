# %%
import os
import glob
import shlex
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import sox


# %%

recordings = Path("/home/mark/tinyspeech_harvard/navajo/NavajoSpeechRecordings")
recordings = recordings.absolute()

keywords = {}
for ix, r in enumerate(os.listdir(recordings)):
    assert not os.path.isdir(recordings / r), "contains subdirectories"
    assert "Navajo" in r, "does not conform to naming structure"
    split = r.find("Navajo")
    keyword = r[:split]
    if not keyword in keywords:
        keywords[keyword] = []
    keywords[keyword].append(r)

# %%

fig, ax = plt.subplots()
ax.bar(keywords.keys(), [len(v) for v in keywords.values()])
ax.set_xticklabels(keywords.keys(), rotation=70)
# ax.set_ylim([0, 3000])
# fig.set_size_inches(40, 10)

# %%
for kw, kwps in keywords.items():
    for p in kwps:
        pth = str(recordings / p)
        duration_s = sox.file_info.duration(pth)
        print(duration_s)
        break

# %%
wanted_words = ["Hello", "Thanks"]
data_dir = Path("/home/mark/tinyspeech_harvard/navajo/ogg_data/")
os.makedirs(data_dir, exist_ok=True)
for kw in wanted_words:
    dest_dir = data_dir / kw.lower()
    os.makedirs(dest_dir, exist_ok=True)
    for fn in keywords[kw]:
        src = recordings / fn
        shutil.copy2(src, dest_dir)

# %%
# https://github.com/tinyMLx/colabs/blob/master/4-6-8-CustomDatasetKWSModel.ipynb
# mkdir wavs
# find *.ogg -print0 | xargs -0 basename -s .ogg | xargs -I {} ffmpeg -i {}.ogg -ar 16000 wavs/{}.wav
# rm -r -f *.ogg
# run els docker container

# %%
# copy in speech_commands data
scdata = Path("/home/mark/tinyspeech_harvard/speech_commands")
for d in os.listdir(scdata):
    if not os.path.isdir(scdata / d):
        continue
    print(d)
    assert d not in ["hello", "thanks"]  # TODO(mmaz) pull this from an above cell
    src = scdata / d
    dest = Path("/home/mark/tinyspeech_harvard/navajo/wav_data") / d
    shutil.copytree(src, dest)
# %%
WANTED_WORDS = "hello,thanks"
number_of_labels = WANTED_WORDS.count(",") + 1
number_of_total_labels = number_of_labels + 2  # for 'silence' and 'unknown' label
equal_percentage_of_training_samples = int(100.0 / (number_of_total_labels))
SILENT_PERCENTAGE = equal_percentage_of_training_samples
UNKNOWN_PERCENTAGE = equal_percentage_of_training_samples
# %%
