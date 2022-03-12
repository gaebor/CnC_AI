# use bigeditor (https://www.ppmsite.com/osbigeditorinfo/) to extract from .MEG files

function parse_name {
    filename="$1"
    sed 's/\([-a-zA-Z0-9]\+\)\(.[A-Z]\+\)\?[-_]\([0-9]\+\)\(-[0-9]\+\)\?/\1\t\3/'<<<"${filename%.*}"
}

dirname=.
if [ "$1" ]
then
    dirname="$1"
fi

(
for file in `find "$dirname" -iname '*.tga' -type f` `find "$dirname" -iname '*.dds' -type f`
do
    parse_name `basename $file`
done

for zip_file in `find "$dirname" -iname '*.zip' -type f`
do
    for file in `unzip -l $zip_file | tail +4 | head -n -2 | sed 's/  \+/\t/g' | cut -f4 | grep -i tga`
    do
        parse_name $file
    done
done
) | sort | uniq | cut -f1 | uniq -c | sort -rn
