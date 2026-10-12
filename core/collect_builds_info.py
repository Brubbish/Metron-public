import os
import json
from collections import defaultdict

OwnerRepoBins = defaultdict(lambda: defaultdict(set))

target1 = "/home/qsh/system_datasets"
for tail in ["", "_2" ,"_php"]:
    targetfolder = f"{target1}/collects_commit_cveid{tail}/bin"
    for target in os.listdir(targetfolder):
        owner, repo = target.split("##")
        for root, dirs, files in os.walk(os.path.join(targetfolder, target)):
            for binname in files:
                if any(binname.endswith(x) for x in ['.i64', '.json', '.o']):
                    continue
                # if file not in OwnerRepoBins[owner][repo]:
                OwnerRepoBins[owner][repo].add(binname)
                
def detect_elf(filepath):
    return open(filepath,"rb").read(4) == b"\x7fELF"

def detect_bins(install_path):
    has_elf = False
    binspath = []
    for root, dirs, files in os.walk(install_path):
        # if has_elf:
        #     break
        for file in files:
            filepath = os.path.join(root, file)
            if os.path.islink(filepath):
                continue
            if detect_elf(filepath):
                has_elf = True  
                binspath.append(filepath)

    return binspath

# target2 = []
for recordfile in ['record_07.txt', 'record_07.txt_2', 'record_new_2.txt']:
    lines = [x.strip() for x in open(recordfile, "r").readlines()]
    for line in lines:
        # /build/unicode-org/icu/990779c4bf935d2741ef6514f4d070231c388d05 timeout
        if not line.endswith(" done"):
            continue
        owner = line.split("/")[-3]
        repo = line.split("/")[-2]
        install_path = line+"/install"
        if not os.path.exists(line):
            continue
        bins = detect_bins(install_path)
        for binname in bins:
            OwnerRepoBins[owner][repo].add(binname)


for owner, repos in OwnerRepoBins.items():
    for repo, bins in repos.items():
        repos[repo] = list(bins)

with open("collect_builds_info.json","w") as f:
    json.dump(OwnerRepoBins, f ,indent=2)
