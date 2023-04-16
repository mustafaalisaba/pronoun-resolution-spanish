from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, AdamW
import torch
import pandas as pd 
import re

def dataset_select(_name, _split):
    """ Returns the text with corresponding label """
    dataset = load_dataset(_name, _split)
    print(dataset)
    # df = dataset.to_pandas()
    # data = df[["sentence", "answer"]]
    # sentences = df["sentence"]
    # labels = df["answer"]
    if _name == "wsc285": # TAREA: Preparar el dataset wsc
        for i in range(0, 285):
            data = dataset[i]
    elif _name in ["hackathon-somos-nlp-2023/winogrande_train_s_spanish", "winogrande"]:
        raw_train_dataset = dataset["train"]
        sentences = raw_train_dataset["sentence"]
        labels = raw_train_dataset["answer"]
        option1 = raw_train_dataset["option1"]
        option2 = raw_train_dataset["option2"]
        i = 0
        new_sentences = []
        new_labels = []
        for s in sentences:
            s1 = re.sub("_", "</s> "+option1[i], s)
            s2 = re.sub("_", "</s> "+option2[i], s)
            new_sentences.append(s1)
            new_sentences.append(s2)
            # print(type(labels[i]))
            if labels[i] == "1":
                l1 = 1
                l2 = 0
            else:
                l1 = 0
                l2 = 1
            if labels[i] == "2":
                l2 = 1
                l1 = 0
            else:
                l2 = 0
                l1 = 1
            new_labels.append(l1)
            new_labels.append(l2)
            # print(s1, labels[i], l1)
            # print(s2, labels[i], l2)
            i += 1
        print(len(new_sentences), len(new_labels))
        # print(sentences,type(sentences))
        
        data = pd.DataFrame({'Sentence' : sentences, 'Label' : labels})
        # print(raw_train_dataset)
        # print(raw_train_dataset['sentence'])
        # print(raw_train_dataset['answer'])
    else:
        print("Please choose a proper name: wsc285 or winogrande_small")
        return
    # data = data.head(640)
    return data

def main():
    """ Execution starts here """
    # wsc_dataset = load_dataset("winograd_wsc", "wsc285") # , split="test"
    # winogrande_dataset = load_dataset("winogrande", "winogrande_s") # , split="train"
    # dataset = load_dataset("hackathon-somos-nlp-2023/winogrande_train_s_spanish", "hackathon-somos-nlp-2023--winogrande_train_s_spanish", split="train")
    # print(dataset)
    # print(dataset[0])
    # print(dataset["train"]["sentence"])
    # print(test_dataset, len(test_dataset))
    # print(train_dataset, len(train_dataset))

    name = "hackathon-somos-nlp-2023/winogrande_train_s_spanish"
    subset = "hackathon-somos-nlp-2023--winogrande_train_s_spanish"
    split = "train"

    # name = "winogrande"
    # subset = "winogrande_s"
    # split = "train"

    split_check = get_dataset_split_names(name)
    configs_check = get_dataset_config_names(name)
    print(split_check)
    print(configs_check)

    raw_data = dataset_select(name, split)

    # print(raw_data)

    # for i in range(0, 284):
    #     test_instances = test_dataset[i]
    #     print(test_instances)
    # train_instances = train_dataset[0]

    # print(train_instances)
    
    checkpoint = "DeepPavlov/roberta-large-winogrande" # TAREA: Probar para el inglés, calcular exactitud
    # batch_size = 16
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    # Ian se ofreció como voluntario para comer el menudo de Dennis después de haber terminado su porción porque </s> Ian despreciaba comer intestino. 
    # Ian se ofreció como voluntario para comer el menudo de Dennis después de haber terminado su porción porque </s> Dennis despreciaba comer intestino.

    txt1 = ["The city councilmen refused the demonstrators a permit because </s> the city councilmen advocated violence"]
    txt2 = ["The city councilmen refused the demonstrators a permit because </s> the demonstrators advocated violence"]

    inputs = tokenizer(txt1, padding="longest", return_tensors="pt")

    # tokenized_dataset = train_dataset.map(tokenize_function, batched=True)

    with torch.no_grad():
        logits = model(**inputs).logits
    
    print(logits)
    
    predicted_class_id = logits.argmax().item()
    ans = model.config.id2label[predicted_class_id]
    
    print(ans)
    
    return 0

if __name__ == '__main__':
    main()