example_file='/home/chamo/Documents/data/openImg/sample_submission.csv'
result_file='/home/chamo/Documents/data/UntitledFolder/expanded_test.csv'
ouput_file='/home/chamo/Documents/data/UntitledFolder/padded_test.csv'

result_dict={}
line_count=1
img_count=0
with open(result_file, "r") as f_result:
    line = f_result.readline()
    while True:
        line = f_result.readline()
        if line=='':
            break
        line_count = line_count + 1
        splited = line.split(",")
        image_name = splited[0]
        if image_name in result_dict.keys():
            print('rep img')
        if len(splited)<2:
            print('error')
        if splited[1] !='\n':
            cache_re=splited[1]
            splited=splited[1].split(' ')
            box_count = 0
            new_str=''
            while True:
                new_str = new_str + splited[box_count * 6 + 0]+' '
                new_str = new_str + splited[box_count * 6 + 1] + ' '
                new_str = new_str + splited[box_count * 6 + 3] + ' '
                new_str = new_str + splited[box_count * 6 + 2] + ' '
                new_str = new_str + splited[box_count * 6 + 5] + ' '
                new_str = new_str + splited[box_count * 6 + 4] + ' '
                box_count = box_count + 1
                if len(splited) - 1 <= box_count * 6:
                    break
            result_dict[image_name]=new_str+'\n'
            img_count=img_count+1
print(img_count)
with open(example_file, "r") as f_example:
    with open(ouput_file, "w") as f_out:
        line = f_example.readline()
        f_out.write(line)
        while line:
            line = f_example.readline()
            if line=='':
                break
            splited = line.split(",")
            image_name = splited[0]
            item_str=image_name+','
            if image_name in result_dict.keys():
                item_str=item_str+result_dict[image_name]
            else:
                item_str = item_str +'\n'
            f_out.write(item_str)








