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
find "$dirname" \( -iname '*\.tga' -o -iname '*\.dds' \) -type f -exec basename {} \; | \
while read file
do
    parse_name $file
done

find "$dirname" -iname '*\.zip' -type f | while read zip_file
do
    unzip -l $zip_file | tail +4 | head -n -2 | sed 's/  \+/\t/g' | cut -f4 | grep -i tga | \
    while read file
    do
        parse_name $file
    done
done
) | sort -r | awk '{ print $2+1 " " $1 }' | uniq -f 1 | sort -nr
