import csv
import os

def merge_annotations(in_folder, out_file):
    tag_lookup={}
    id_lookup={}
    for file in sorted(os.listdir(in_folder)):
        print(file)
        with open(in_folder+"/"+file, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in reader:
                tag=row[0]
                id=row[2]
                ignore=False
                try:
                    val = float(id)
                except ValueError:
                    ignore=True
                    pass

                if ignore:
                    continue
                content=row[1]
                tag_lookup[content]=tag
                id_lookup[content]=id

    with open(out_file, 'w', newline='',encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for key, value in id_lookup.items():
            tag=tag_lookup[key]
            writer.writerow([tag,key, value])



# in_folder="/home/zqz/Work/chase/data/annotation/unfiltered"
# out_file="/home/zqz/Work/chase/data/annotation/unfilered_merged.csv"
# in_folder="/home/zqz/Work/chase/data/annotation/keyword_filtered"
# out_file="/home/zqz/Work/chase/data/annotation/keywordfilered_merged.csv"
print(os.getcwd())
in_folder="../../../data/annotation/tag_filtered"
out_file="../../../data/annotation/tagfilered_merged.csv"
merge_annotations(in_folder,out_file)
