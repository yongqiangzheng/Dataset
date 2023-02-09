# ABSAdataset


## Introduction

**xml_original** is download from SemEval official website.

When we use **xml_original** to obtain the dataset, we find that it is inconsistent with other existing ABSA work.

For a fair comparison, we manually revise the official dataset and obtain **xml_v2**.

After that, we use spaCy and adopt some regex rules to tokenize the sentence to generate **one_sentence_one_aspect**

Finally, we align input sentences with different aspects in the same sentence to generate **one_sentence_all_aspect**

The file format as follows:

### one_sentence_one_aspect

```
I charge it at night and skip taking the $T$ with me because of the good battery life .
cord
0
I charge it at night and skip taking the cord with me because of the good $T$ .
battery life
1
```

### one_sentence_all_aspect

```
I charge it at night and skip taking the cord with me because of the good battery life .
cord#9#10 /// battery life#16#18 ///
0 1
```

## process new dataset

```
sh create_dir.sh
```

If you want to process your dataset by yourself, you can use following code in main.py

We take the SemEval2014 Laptops dataset as an example:

```
lap14_train = Dataset4ABSA('lap14', 'train')
data = lap14_train.parse_semeval14()
lap14_train.write_dataset(data, 'one')
```

Note: Please ensure that the above code will not output sentence pairs, otherwise it means that there are still unaligned sentences, which will cause problems when generating **one_sentence_all_aspect**

```
lap14_train.write_dataset(data, 'all')
```