import json
# import ramdom

# eval_data_path = '/data/bob/synthetic/2021-04-22 4m/1m_synthetic_cap.json'
# data_length = 1000

def get_data(data_path, data_num):


    with open(data_path, 'r') as file:
        real_data = json.loads(file.read())[:data_num]

    real_data[:2]
    input_data = []
    real_labels_data = []

    # real_data[0]
    for i in range(len(real_data)):
        # i = 0
        # print(i)
        example = real_data[i]
        if 'labels' in example.keys():
            real_labels = example['labels']
            new_description = []
            new_label = []
            for j in real_labels:
                # print(j)
                # 如果不是others
                if isinstance(j, str):
                    new_description.append(example[j].strip())
                    new_label.append(j)

            new_description = ' '.join(new_description)
            input_data.append(new_description)
            real_labels_data.append(new_label)

    return (input_data, real_labels_data)

def modified_data(input_data):
    input_data_modified = []
    for data in input_data:
        # input_data = data.rstrip()
        input_data_modified.append(data.replace('(', '').replace(')', ''))
    return input_data_modified