#!/bin/bash  
  
# 遍历当前目录下所有的.gjf文件  
for file in *.gjf; do  
    # 检查文件是否存在  
    if [ -f "$file" ]; then  
        # 使用g16命令执行当前文件  
        # 注意：你可能需要根据g16的具体用法调整下面的命令  
        # 假设g16可以直接这样调用文件，或者你可能需要指定更复杂的命令行参数  
        g16 "$file"  
          
        # 检查上一个命令是否成功执行  
        if [ $? -ne 0 ]; then  
            echo "Error executing $file with g16"  
            # 可以选择在这里退出脚本，或者继续执行下一个文件  
            # exit 1  
        fi  
    else  
        echo "Warning: $file does not exist."  
    fi  
done  
  
echo "All .gjf files have been processed."
