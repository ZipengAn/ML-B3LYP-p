#!/bin/bash  
  
# 指定目录，这里假设是当前目录  
dir="./"  
  
# 遍历目录下的所有.xyz文件  
for file in "$dir"*.xyz; do  
    # 提取文件名（不包括路径和扩展名）  
    filename=$(basename -- "$file" .xyz)  
      
    # 使用printf进行格式化，%03d表示至少三位数字，不足的前面补零  
    formatted_filename=$(printf "%03d" "$filename")  
      
    # 构造新的文件名（包括扩展名）  
    new_file="$dir$formatted_filename.xyz"  
      
    # 如果新文件名与旧文件名不同，则重命名文件  
    if [ "$file" != "$new_file" ]; then  
        mv "$file" "$new_file"  
        echo "Renamed $file to $new_file"  
    fi  
done
