line=$1
files="*.jpg"
file=$line$files
echo $file
for filepath in ${file}

do
  ./cat_avatar ${filepath} >>file.txt
done