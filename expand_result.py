import argparse
import json

hierarchy_file = '/home/chamo/Documents/work/OpenImgChamo/config/bbox_labels_500_hierarchy.json'
result_file = '/home/chamo/Documents/data/UntitledFolder/test.csv'
ouput_file = '/home/chamo/Documents/data/UntitledFolder/expanded_test.csv'
ouput_box_file = '/home/chamo/Documents/data/UntitledFolder/box_test.csv'

result_dict = {}
line_count = 1
img_count = 0


def _update_dict(initial_dict, update):
    for key, value_list in update.items():
        if key in initial_dict:
            initial_dict[key].extend(value_list)
        else:
            initial_dict[key] = value_list


def _build_plain_hierarchy(hierarchy, skip_root=False):
    all_children = []
    all_keyed_parent = {}
    all_keyed_child = {}
    if 'Subcategory' in hierarchy:
        for node in hierarchy['Subcategory']:
            keyed_parent, keyed_child, children = _build_plain_hierarchy(node)
            _update_dict(all_keyed_parent, keyed_parent)
            _update_dict(all_keyed_child, keyed_child)
            all_children.extend(children)

    if not skip_root:
        all_keyed_parent[hierarchy['LabelName']] = all_children
        all_children = [hierarchy['LabelName']] + all_children
        for child, _ in all_keyed_child.items():
            all_keyed_child[child].append(hierarchy['LabelName'])
        all_keyed_child[hierarchy['LabelName']] = []

    return all_keyed_parent, all_keyed_child, all_children


with open(hierarchy_file) as f:
    hierarchy = json.load(f)
    all_keyed_parent, all_keyed_child, all_children = _build_plain_hierarchy(hierarchy, skip_root=True)
    with open(result_file, "r") as f_result:
        with open(ouput_file, "w") as f_out:
            with open(ouput_box_file, "w") as f_box_out:
                line = f_result.readline()
                f_out.write(line)
                total_add_count=0
                f_box_out.write(
                    'ImageID,Source,LabelName,Score,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside')
                while True:
                    line = f_result.readline()
                    if line == '':
                        break
                    line_count = line_count + 1
                    splited = line.split(",")
                    image_name = splited[0]
                    if image_name=='':
                        break
                    if image_name in result_dict.keys():
                        print('rep img')
                    if len(splited) < 2:
                        print('error')
                    new_str = []
                    if splited[1] != '\n':
                        cache_re = splited[1]
                        splited = splited[1].split(' ')
                        box_count = 0
                        boxes_list=[]
                        class_list=[]
                        while True:
                            box_info = []
                            for cc in range(5):
                                temp_index=box_count * 6 + cc+1
                                temp_val=float(splited[temp_index])
                                if temp_val>1:
                                    splited[temp_index]='1.0'
                                if temp_val<0:
                                    splited[temp_index]='0.0'
                            box_info.append(splited[box_count * 6 + 0]) #class name
                            box_info.append(splited[box_count * 6 + 1]) #confidence
                            box_info.append(splited[box_count * 6 + 2]) #xmin
                            box_info.append(splited[box_count * 6 + 3]) #ymin
                            box_info.append(splited[box_count * 6 + 4]) #xmax
                            box_info.append(splited[box_count * 6 + 5]) #ymax
                            box_count = box_count + 1
                            if len(splited) - 1 <= box_count * 6:
                                break
                            if float(box_info[1])<0.0001:
                                continue
                            if float(box_info[4])-float(box_info[2])<=0 or float(box_info[5])-float(box_info[3])<=0:
                                print('zero size box')
                                continue
                            boxes_list.append(box_info)
                            class_list.append(box_info[0])
                        for box_info in boxes_list:
                            box_box = []
                            box_box.append(image_name)
                            box_box.append('freeform')
                            box_box.append(box_info[0])
                            box_box.append(box_info[1])
                            box_box.append(box_info[2])
                            box_box.append(box_info[4])
                            box_box.append(box_info[3])
                            box_box.append(box_info[5])
                            f_box_out.write(','.join(box_box))
                            f_box_out.write('\n')
                            result =[' '.join(box_info)]
                            assert box_info[0] in all_keyed_child
                            parent_nodes = all_keyed_child[box_info[0]]
                            for parent_node in parent_nodes:
                                if parent_node in class_list:
                                    continue
                                total_add_count=total_add_count+1
                                box_info[0] = parent_node
                                box_box=[]
                                box_box.append(image_name)
                                box_box.append('freeform')
                                box_box.append(box_info[0])
                                box_box.append(box_info[1])
                                box_box.append(box_info[2])
                                box_box.append(box_info[4])
                                box_box.append(box_info[3])
                                box_box.append(box_info[5])
                                f_box_out.write(','.join(box_box))
                                f_box_out.write('\n')
                                result.append(' '.join(box_info))
                                #print(image_name)
                            new_str.append(' '.join(result))
                            img_count = img_count + 1
                    f_out.write(image_name+','+' '.join(new_str)+'\n')
print(total_add_count)
