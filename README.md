# Automated Text Summarization Using BERT: A Case Study on Indonesian News Articles with the Liputan6 Dataset

## Introduction
Text Summarization is a computational technique designed to condense lengthy documents into concise summaries while retaining the most important information. This method helps streamline the process of extracting key insights from large volumes of text, saving time and effort for individuals who need to quickly comprehend important documents. The advancement of natural language processing (NLP) techniques, such as transformer models like BERT, has significantly improved the accuracy and efficiency of automated summarization systems.

In this project, we aim to develop an automated text summarization system using the BERT model, which is pre-trained on large corpora of text to generate coherent and informative summaries. The system is designed to assist business executives, researchers, and professionals by reducing the amount of time spent reading and processing complex documents.

## Objective
The primary objective of this project is to experiment with a BERT-based model to perform text summarization on a dataset of news articles. The system will automatically generate concise summaries of the original text, maintaining the essential information while significantly reducing the text length. 

## Dataset
### Dataset Description
The Liputan6 Dataset is a collection of Indonesian news articles sourced from the Liputan6 website. It serves as the primary resource for training and evaluating text summarization models in the Indonesian language.
### Record Structure
Each record in the dataset contains the following fields:
1. id: A unique identifier for each article.
2. url: The web link to the news article.
3. clean_article: The full content of the news article, which has been preprocessed to remove unnecessary elements such as advertisements or formatting tags.
4. clean_summary: An abstractive summary of the article, generated either manually or automatically.
5. extractive_summary: Index-based extractive summaries, which highlight key sentences from the original text.

### Dataset Subsets
The project relies on the id_liputan6 dataset, which is available on the Hugging Face platform. This dataset, specifically curated from Liputan6 — a major Indonesian news source — provides rich text-summarization pairs. These pairs are crucial for training models that can perform text summarization in the Indonesian language.
The id_liputan6 dataset is divided into two main subsets:

1. Xtreme Set:
        a. Designed for challenging summarization tasks
        b. Contains diverse article lengths and complexities
        c. Used to test model robustness in complex scenarios

<img width="1077" alt="Screen Shot 2024-10-19 at 12 17 07" src="https://github.com/user-attachments/assets/1bcaff02-0644-4a59-ac16-8794f61405d3">

2. Canonical Set:
        a. Primary dataset for training and benchmarking
        b. Larger collection of news articles and summaries
        c. Balanced representation of various categories (politics, entertainment, sports, etc.)
        d. Reflects diversity of mainstream Indonesian news reporting
        e. Covers a comprehensive range of topics and writing styles

<img width="1017" alt="Screen Shot 2024-10-19 at 12 24 17" src="https://github.com/user-attachments/assets/62ea647d-0b2f-4d32-ad0e-e826e260f880">

### Importance for the Project
- The Canonical Subset is used for fine-tuning the Bert2GPT model
- Exposes the model to a wide variety of news content
- Ensures the model can generate accurate, concise, and coherent summaries
- Allows the model to learn language nuances and adapt to different contexts
- Makes the model highly relevant and applicable across different types of Indonesian news articles

## Methodology
### Exploratory Data Analysis (EDA)
The Exploratory Data Analysis (EDA) phase is critical in understanding the dataset's underlying patterns, structure, and characteristics. In this project, EDA offers insights into the id_liputan6 dataset, guiding the development and fine-tuning of the Bert2GPT Indonesian Text Summarizer. The following analytical steps were taken to provide a comprehensive understanding of the dataset:

#### 1. Data Cleaning, Summary and Insights


1. Text Cleaning:  The cleaning process effectively prepares the text for analysis:
- Lowercase Conversion: All text is converted to lowercase, ensuring uniformity.
- Removal of Unwanted Text: Instances of "Liputan6.com," location prefixes, and specific text in parentheses or brackets are removed to focus on the core content.
- Special Character Removal: Non-alphanumeric characters are eliminated, which simplifies the text and helps in further analysis.
- Trimming Excess Spaces: Any extra spaces in the text are removed to maintain clean formatting.

2. Article and Summary Statistics:
        a. Cleaned Article Statistics:
                - Average length: 181.524 words
                - Maximum length: 1064 words
                - Minimum length: 68 words
        b. Cleaned Summary Statistics:
                - Average length: 23.127 words
                - Maximum length: 37 words
                - Minimum length: 13 words
           Insights:
                - There's a significant difference between article and summary lengths.
                - The compression ratio is about 7.8:1 (181.524 / 23.127), meaning summaries are typically about 12.7% the length of the original articles.
                - Articles have a wide range of lengths (68 to 1064 words), which might require attention during model training.
                - Summaries are more consistent in length (13 to 37 words), which is good for generating predictions.

3. Word Frequency Analysis:
Top Words in Cleaned Articles:
`Most frequent words include: di, yang, dan, itu, ini, jakarta, dari, untuk, dengan, dalam.`
Top Words in Cleaned Summaries:
`Most frequent words include: di, dan, yang, akan, tak, harga, untuk, pemerintah, dari, sejumlah.`

<img width="1336" alt="Screen Shot 2024-10-22 at 21 47 55" src="https://github.com/user-attachments/assets/caca3868-029c-4c4e-9b9c-e5f08cd64d6c">



#### 2. Text Preprocessing
The preprocessing step seems to have worked, but there are a few observations:

- The lemmatization doesn't appear to be working as expected for Indonesian. This is likely because NLTK's WordNetLemmatizer is designed for English.

- Stop words are being removed, but some common words (like "yang", "di", "dan") are still present. This suggests that the Indonesian stop words list might not be comprehensive.

- Punctuation marks are still present in the processed text. You might want to remove these in a future iteration.
<img width="1440" alt="Screen Shot 2024-10-19 at 12 59 20" src="https://github.com/user-attachments/assets/620733ab-ed1e-4070-8987-4a09d5ee5295">

- Word Cloud Visualization: Create word clouds for both original and processed texts to visually compare the most prominent words.

![a](https://github.com/user-attachments/assets/4db8cb1b-50c3-4f88-ae1b-56154a64ebbe)

![b](https://github.com/user-attachments/assets/2fbef44b-d227-41b2-a5a9-efd0fb8114f1)

- Unique Words Count: Compare the number of unique words in the original and processed datasets to see how much the vocabulary has been reduced.
          a. Original article vocabulary size: 612672
          b. Processed article vocabulary size: 269388
          c. Original summary vocabulary size: 172071
          d. Processed summary vocabulary size: 93090


#### 3. Distribution of Article Lengths
This step examines the lengths of processed articles to understand the overall distribution within the dataset. By plotting a histogram, we can observe variations in article length and detect potential outliers or biases in the data. This analysis helps determine if certain article lengths dominate the dataset, which may impact the model’s performance.

![c](https://github.com/user-attachments/assets/148b24cb-88cc-4737-9cf5-482d09a5c763)

The distribution of article lengths shows a notable concentration of shorter articles, with 16.31% having fewer than 500 characters. The mean article length is approximately 885.24 characters, while the median is 744 characters, indicating that most articles fall within the range of 562 to 1,048 characters. The dataset reveals a minimum length of 29 characters and a maximum length of 28,678 characters, with only 267 articles (0.13%) exceeding 5,000 characters. The skewness of 4.28 and kurtosis of 66.28 reflect a strong positive skew, suggesting that while most articles are relatively concise, there are a few significant outliers that are considerably longer. These findings underscore the need for a summarization model capable of effectively managing a diverse range of article lengths, from very short to exceptionally long.

#### 4. Distribution of Summary Lengths
Similarly, I analyze the distribution of summary lengths to understand the patterns in the summarization process. This step is essential to ensure that the generated summaries are concise while still retaining key information. A histogram is used to visualize how summary lengths vary, aiding in fine-tuning the model to produce balanced and informative summaries.

![d](https://github.com/user-attachments/assets/14628c1b-b26f-45dd-a007-d4206d98696e)

The distribution of summary lengths indicates a significant concentration of shorter summaries, with 0.08% having fewer than 50 characters. The mean summary length is approximately 138.88 characters, while the median is 138 characters, suggesting that most summaries fall within the range of 119 to 156 characters. The dataset reveals a minimum length of 26 characters and a maximum length of 443 characters, with no summaries exceeding 500 characters.

The skewness of 1.21 and kurtosis of 6.46 reflect a positive skew, indicating that while most summaries are concise, there are a few longer summaries that stretch the distribution. These findings highlight the importance of designing a summarization model capable of effectively handling a consistent range of summary lengths, which are predominantly short, with very few outliers.


#### 5. Most Frequent Terms in Articles
This step identifies the most frequent terms in the original articles. By visualizing these terms through a bar plot, we gain insight into the common themes and topics that dominate the dataset. This information helps in understanding the general context of the articles and aligning the model to prioritize relevant terms during summarization.

![e](https://github.com/user-attachments/assets/c8adad3c-43aa-496f-b12d-18bec03cd4b4)

<b>Top 5 Terms:</b>
1. jakarta: 180155
2. warga: 126085
3. rumah: 97624
4. polisi: 88010
5. indonesia: 82921


#### 6. Most Frequent Terms in Summaries
This analysis mirrors the frequent term analysis for articles but focuses on the summaries. The frequent terms in the summaries provide insights into which parts of the articles are being prioritized in the summarization process. A bar plot is used to showcase the top terms, which help in refining the model to generate meaningful and focused summaries.

![f](https://github.com/user-attachments/assets/4ef97454-c056-4712-82e1-cba2eeb7d974)

The analysis of the Most Frequent Terms in Summaries provides a view of the most common words used in the summaries of the articles. Below are the key insights based on the top 30 terms:

<b>Top 5 Terms:</b>
1. warga: 24924
2. jakarta: 16185
3. rumah: 15867
4. korban: 14432
5. indonesia: 13673


#### 7. Scatter Plot for Article vs. Summary Length
A scatter plot is used to compare the lengths of the original articles and their corresponding summaries. This visualization provides an overview of the relationship between the article length and the compression ratio during summarization. It helps in analyzing how well the model balances between retaining information and reducing text length.

![g](https://github.com/user-attachments/assets/10ffb3bf-096f-4841-bf45-97ad2131176e)

The Scatter Plot for Article vs. Summary Length provides insights into the relationship between the length of the articles and their corresponding summaries. Here are the key observations:

The weak correlation and consistent summary lengths suggest that the summarization process is effective in compressing information regardless of article length. This ensures that even longer articles can be summarized succinctly, while shorter articles are not excessively truncated. The average compression ratio of 0.19 further emphasizes that the summaries are concise, significantly reducing the original article lengths while retaining core content.

#### 8. Box Plot for Distribution of Text Lengths
The box plot offers a visual representation of the variance and central tendencies in text lengths for both articles and summaries. This helps to understand the range of lengths the model will need to handle and ensures that outliers or excessively long articles do not skew the model's training.

![H](https://github.com/user-attachments/assets/0ae67d52-71ea-4f44-815d-e5a0bb9d88bb)

The box plot clearly illustrates that summaries are significantly shorter than the original articles, showcasing a more consistent length distribution. Articles exhibit a wider range of lengths, including a few very long outliers, while summaries typically remain concise and within a narrow range. This visual representation reinforces the effectiveness of the summarization process in condensing text significantly while preserving essential information.


#### 9. Plotting Bi-grams in Articles
Bi-grams, which are pairs of consecutive words, are plotted to reveal common phrase patterns in the articles. Analyzing bi-grams helps understand the contextual relationships between words and provides insights into frequently occurring phrases that the model can prioritize during summarization.

![i](https://github.com/user-attachments/assets/6f6f3332-0745-404e-bef1-8cd0d347d252)


#### 10. Plotting Bi-grams in Summaries
Bi-gram analysis is extended to summaries to uncover the most common two-word sequences. This step helps in refining the model’s ability to capture and reproduce key phrases in the summaries that are crucial for maintaining coherence and context.

![j](https://github.com/user-attachments/assets/78b87433-d97b-4296-bbed-63dec2fe1ba2)


#### 11. Plotting Tri-grams in Articles
The tri-gram analysis focuses on frequent three-word sequences in the articles. By identifying these tri-grams, we gain deeper insights into contextual word relationships, which can help the model generate more accurate and coherent summaries.

![k](https://github.com/user-attachments/assets/c7bee130-ad73-487e-be3b-41a30348962f)


#### 12. Plotting Tri-grams in Summaries
As with the articles, tri-gram analysis is applied to the summaries to understand which three-word phrases are most commonly retained in the summarization process. This informs the model on how to reproduce important multi-word phrases that carry significant meaning.

![l](https://github.com/user-attachments/assets/5ffb98b2-fb72-4c1a-a752-04f7e968a67d)


#### 13. Sentiment Analysis for Articles
Sentiment analysis is conducted on the original articles to determine the overall emotional tone and sentiment polarity (positive, negative, or neutral). Understanding the sentiment of the articles helps ensure that the summarization process maintains the original emotional context and tone.

![m](https://github.com/user-attachments/assets/d567453c-1af2-48a6-a799-218cabc56dde)

The histogram illustrates that the vast majority of articles cluster around a sentiment polarity of 0.00, highlighting a strong prevalence of neutral sentiments. There are very few articles with extreme positive or negative sentiments, as indicated by the sparse distribution in those areas.
The findings suggest that the articles are primarily neutral in sentiment, with a minority expressing positive or negative sentiments. This may indicate that the content covered in the articles tends to be factual or informational rather than emotionally charged. The analysis underscores the importance of understanding sentiment for tasks such as summarization, as it may impact the interpretation of the articles and their overall tone.


#### 14. Sentiment Analysis for Summaries
Extending sentiment analysis to the summaries ensures that the generated summaries preserve the sentiment of the original articles. This step is important for maintaining the integrity and emotional tone of the content.

![n](https://github.com/user-attachments/assets/574ddae3-ef62-4f97-8550-c323373f6a64)

The histogram illustrates that the vast majority of summaries cluster around a sentiment polarity of 0.00, emphasizing a strong dominance of neutral sentiments. Very few summaries exhibit extreme positive or negative sentiments, as indicated by the sparse distribution in those areas.
The findings suggest that the summaries are primarily neutral in sentiment, with a minimal proportion expressing positive or negative sentiments. This may indicate that the summarization process tends to retain a balanced tone, likely focusing on factual information rather than emotional expression. The analysis highlights the importance of sentiment in understanding the overall tone of the summaries, which is crucial for applications such as content recommendation and audience engagement. ​​


#### 15. Top Words in Topics
Topic modeling, such as Latent Dirichlet Allocation (LDA), is used to uncover the main themes and topics within the dataset. By identifying the top words in various topics, we can understand the content diversity and ensure the model is well-tuned to represent the range of subjects present in the news articles.
Here are the top words found in the five key topics:

`Top words in LDA topics:
Topic 1: jakarta, presiden, indonesia, ketua, negara, partai, anggota, pemerintah, menteri, aceh
Topic 2: warga, air, banjir, rumah, jalan, desa, jawa, kabupaten, suara, korban
Topic 3: rp, jakarta, harga, anak, pemerintah, minyak, indonesia, ribu, persen, pasar
Topic 4: pemain, tim, pertandingan, musim, gol, klub, menit, liga, bermain, babak
Topic 5: polisi, korban, rumah, warga, kepolisian, jakarta, tersangka, kota, jawa, orang

Compression ratio: 0.16`

The analysis of the top words in LDA topics illustrates the diverse range of subjects covered in the articles, from politics and community issues to economics and sports. The low compression ratio signifies that the summaries effectively capture the essence of the articles, making them concise yet informative. Understanding these topics can aid in tailoring content delivery, enhancing engagement, and supporting more targeted information dissemination strategies.


## Data Preprocessing
### 1. Splitting the Dataset
The dataset is divided into training (79,413 examples) and validation (19,854 examples) sets. This separation ensures the model can be evaluated on unseen data for better generalization.
### 2. Loading the Tokenizer from a Pretrained Model
The tokenizer from the pretrained model "cahya/bert2gpt-indonesian-summarization" is employed, with special tokens for marking the start and end of sequences.

<img width="1081" alt="Screen Shot 2024-10-19 at 14 52 04" src="https://github.com/user-attachments/assets/711fd77a-0222-4d32-b15e-be37327ce7db">

### 3. Defining the Maximum Length of the Input and Target Sequences
Both the input (articles) and target (summaries) sequences are limited to a maximum length of 256 tokens for efficient batching and uniformity.
### 4. Tokenizing and Encoding Articles and Summaries
A preprocessing function tokenizes and encodes the articles and summaries, ensuring they are padded or truncated to the specified length.
### 5. Mapping Train and Validation Dataset
The preprocessing function is applied to both the training and validation sets, resulting in tokenized, encoded, and structured datasets with input IDs, attention masks, and decoder input IDs ready for model input.

<img width="1100" alt="Screen Shot 2024-10-22 at 22 25 56" src="https://github.com/user-attachments/assets/371f137f-02d5-4201-8037-ab062d0e3904">

### 6. Processed Example
For verification, processed examples from both datasets (training and validation datasets) are printed, showcasing the encoded values (input_ids, attention_mask, labels, decoder_input_ids).
#### Print Processed Examples from the Training Dataset

```"html"
Processed Examples from the Training Dataset:
Example 1:
input_ids: [3, 17715, 1050, 17, 3036, 15, 2647, 29, 11835, 8536, 13630, 2862, 11, 23581, 1006, 12, 3144, 4071, 4510, 17, 6610, 20375, 30044, 1007, 15, 5598, 4643, 1508, 15, 1509, 2167, 1566, 2221, 1495, 1885, 2162, 7058, 1942, 16, 1942, 1510, 12330, 1495, 2177, 3541, 1978, 2042, 17695, 15, 2647] ...
attention_mask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ...
labels: [3, 6610, 20375, 30044, 1007, 15, 5598, 4643, 1508, 15, 1509, 2167, 30851, 1495, 25823, 17695, 1748, 22355, 23581, 1006, 17, 1977, 12188, 2720, 15, 1975, 2757, 1695, 1495, 3030, 4968, 9509, 7109, 15, 10980, 3642, 15, 2165, 2295, 2273, 13884, 16109, 2049, 17, 1, 2, 2, 2, 2, 2] ...
decoder_input_ids: [3, 3, 6610, 20375, 30044, 1007, 15, 5598, 4643, 1508, 15, 1509, 2167, 30851, 1495, 25823, 17695, 1748, 22355, 23581, 1006, 17, 1977, 12188, 2720, 15, 1975, 2757, 1695, 1495, 3030, 4968, 9509, 7109, 15, 10980, 3642, 15, 2165, 2295, 2273, 13884, 16109, 2049, 17, 1, 2, 2, 2, 2] ...


Example 2:
input_ids: [3, 17715, 1050, 17, 3036, 15, 12516, 29, 8492, 2779, 16, 1859, 8707, 3959, 22356, 1967, 1942, 15, 9436, 5, 5112, 1542, 2815, 28676, 17, 29637, 15, 3381, 6995, 5624, 1566, 7244, 1510, 2059, 13200, 2229, 2236, 25682, 1509, 3274, 17, 2199, 1609, 25682, 1520, 7919, 17, 5927, 2957, 1510] ...
attention_mask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ...
labels: [3, 8263, 10210, 3357, 1509, 1852, 1686, 1029, 7364, 5203, 1941, 4005, 20498, 1508, 17, 1753, 12020, 15, 4510, 2401, 2071, 1028, 1509, 17887, 1966, 4865, 17, 10966, 1538, 3598, 10762, 1495, 1688, 6512, 1928, 1560, 3825, 20273, 17, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2] ...
decoder_input_ids: [3, 3, 8263, 10210, 3357, 1509, 1852, 1686, 1029, 7364, 5203, 1941, 4005, 20498, 1508, 17, 1753, 12020, 15, 4510, 2401, 2071, 1028, 1509, 17887, 1966, 4865, 17, 10966, 1538, 3598, 10762, 1495, 1688, 6512, 1928, 1560, 3825, 20273, 17, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2] ...


Example 3:
input_ids: [3, 17715, 1050, 17, 3036, 15, 5138, 29, 4693, 15307, 2225, 2878, 2192, 1495, 5138, 15, 2326, 2341, 15, 12757, 11, 3663, 18, 2178, 12, 15, 14507, 1010, 3359, 4596, 17, 1891, 3473, 2192, 2781, 8325, 61, 1495, 2384, 17249, 16, 17249, 17, 4342, 16, 4342, 1677, 24769, 15069, 2356] ...
attention_mask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ...
labels: [3, 1855, 4446, 6990, 1677, 24769, 1748, 2356, 6200, 1878, 29072, 2054, 17, 2878, 2192, 1495, 5138, 15, 2326, 2099, 15, 1510, 14507, 1010, 3359, 4596, 1885, 1716, 4014, 8325, 61, 2041, 4014, 1509, 4228, 1630, 16, 2614, 9133, 1503, 17, 1, 2, 2, 2, 2, 2, 2, 2, 2] ...
decoder_input_ids: [3, 3, 1855, 4446, 6990, 1677, 24769, 1748, 2356, 6200, 1878, 29072, 2054, 17, 2878, 2192, 1495, 5138, 15, 2326, 2099, 15, 1510, 14507, 1010, 3359, 4596, 1885, 1716, 4014, 8325, 61, 2041, 4014, 1509, 4228, 1630, 16, 2614, 9133, 1503, 17, 1, 2, 2, 2, 2, 2, 2, 2] ...
```

#### Print Processed Examples from Validation Dataset
```"html"
Processed Examples from the Validation Dataset:
Example 1:
input_ids: [3, 16031, 4715, 3028, 455, 6011, 456, 28576, 11832, 1012, 17, 1788, 2627, 3056, 13392, 23774, 1786, 3991, 19881, 11, 2163, 17651, 12, 1795, 16, 8286, 1533, 25321, 2649, 1800, 11037, 1731, 26945, 13958, 3228, 3444, 16, 3444, 5206, 1570, 4152, 12100, 1495, 4863, 6324, 3069, 15, 3981, 1542, 17] ...
attention_mask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ...
labels: [3, 2165, 12020, 1753, 12680, 18397, 15364, 6266, 15, 5905, 10592, 19319, 26208, 2451, 1049, 20199, 2878, 2627, 1572, 7116, 12448, 14836, 17, 5031, 12621, 5995, 8868, 1583, 15, 1891, 1723, 2627, 13392, 1510, 12656, 29357, 1015, 2715, 1495, 3297, 10723, 41, 1637, 6427, 1743, 26208, 2451, 1049, 17, 1] ...
decoder_input_ids: [3, 3, 2165, 12020, 1753, 12680, 18397, 15364, 6266, 15, 5905, 10592, 19319, 26208, 2451, 1049, 20199, 2878, 2627, 1572, 7116, 12448, 14836, 17, 5031, 12621, 5995, 8868, 1583, 15, 1891, 1723, 2627, 13392, 1510, 12656, 29357, 1015, 2715, 1495, 3297, 10723, 41, 1637, 6427, 1743, 26208, 2451, 1049, 17] ...


Example 2:
input_ids: [3, 17715, 1050, 17, 3036, 15, 12516, 29, 17848, 7558, 1495, 2198, 12516, 15, 2326, 2102, 15, 8313, 2815, 1859, 1737, 2340, 17, 1956, 15, 7558, 12528, 9563, 1510, 2340, 1535, 2091, 2947, 1675, 1637, 7558, 2997, 1570, 2573, 7558, 1495, 12516, 17, 5154, 15, 3262, 2653, 2168, 1495, 2042] ...
attention_mask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ...
labels: [3, 7558, 12528, 9563, 1495, 12516, 15, 20565, 15, 4715, 2129, 2878, 3085, 4764, 21217, 1549, 10774, 9173, 6488, 17848, 17, 1956, 15, 17994, 1543, 7483, 1510, 2751, 1789, 4394, 1012, 1495, 2968, 19763, 3590, 1016, 2172, 14223, 3597, 17, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2] ...
decoder_input_ids: [3, 3, 7558, 12528, 9563, 1495, 12516, 15, 20565, 15, 4715, 2129, 2878, 3085, 4764, 21217, 1549, 10774, 9173, 6488, 17848, 17, 1956, 15, 17994, 1543, 7483, 1510, 2751, 1789, 4394, 1012, 1495, 2968, 19763, 3590, 1016, 2172, 14223, 3597, 17, 1, 2, 2, 2, 2, 2, 2, 2, 2] ...


Example 3:
input_ids: [3, 17715, 1050, 17, 3036, 15, 2647, 29, 3162, 1978, 3076, 4355, 2480, 11, 17718, 12, 2618, 12079, 52, 17358, 22138, 10230, 1786, 5924, 9466, 2618, 12079, 1495, 4430, 16, 2602, 1572, 4114, 3641, 8139, 18187, 5369, 2070, 6834, 3101, 18187, 1944, 15, 1971, 5369, 2070, 3434, 17, 3290, 5668] ...
attention_mask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ...
labels: [3, 3162, 17718, 2618, 12079, 17358, 22138, 10230, 1786, 5924, 9466, 2618, 12079, 1495, 4430, 16, 2602, 1572, 4114, 3641, 8139, 18187, 5369, 2070, 6834, 3101, 18187, 1944, 15, 1971, 5369, 2070, 3434, 17, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2] ...
decoder_input_ids: [3, 3, 3162, 17718, 2618, 12079, 17358, 22138, 10230, 1786, 5924, 9466, 2618, 12079, 1495, 4430, 16, 2602, 1572, 4114, 3641, 8139, 18187, 5369, 2070, 6834, 3101, 18187, 1944, 15, 1971, 5369, 2070, 3434, 17, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2] ...
```

#### Printing the Length of Training and Validation Datasets
The lengths of the training and validation datasets give us an overview of the dataset size and the number of examples available for model training and validation:

1. Length of Training Dataset: 155,106 examples.
2. Length of Validation Dataset: 38,777 examples.

This indicates that the training dataset consists of 79,413 data points, which are used to train the machine learning model, while the validation dataset contains 19,854 data points, which are used to evaluate the model's performance during the training process.

These dataset sizes reflect the amount of data available for learning patterns (training) and for assessing generalization (validation).

#### Print The First Element of Training and Validation Dataset
Inspecting the first element of both the training and validation datasets provides insight into the data structure and preprocessing applied before training the model. These details include the processed text, summaries, and tokenized inputs necessary for the Bert2GPT model to learn and make predictions effectively.
```"html"

First Element of Training Dataset:
input_ids: [3, 17715, 1050, 17, 3036, 15, 2647, 29, 11835, 8536, 13630, 2862, 11, 23581, 1006, 12, 3144, 4071, 4510, 17, 6610, 20375, 30044, 1007, 15, 5598, 4643, 1508, 15, 1509, 2167, 1566, 2221, 1495, 1885, 2162, 7058, 1942, 16, 1942, 1510, 12330, 1495, 2177, 3541, 1978, 2042, 17695, 15, 2647]...
token_type_ids: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]...
attention_mask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]...
labels: [3, 6610, 20375, 30044, 1007, 15, 5598, 4643, 1508, 15, 1509, 2167, 30851, 1495, 25823, 17695, 1748, 22355, 23581, 1006, 17, 1977, 12188, 2720, 15, 1975, 2757, 1695, 1495, 3030, 4968, 9509, 7109, 15, 10980, 3642, 15, 2165, 2295, 2273, 13884, 16109, 2049, 17, 1, 2, 2, 2, 2, 2]...
decoder_input_ids: [3, 3, 6610, 20375, 30044, 1007, 15, 5598, 4643, 1508, 15, 1509, 2167, 30851, 1495, 25823, 17695, 1748, 22355, 23581, 1006, 17, 1977, 12188, 2720, 15, 1975, 2757, 1695, 1495, 3030, 4968, 9509, 7109, 15, 10980, 3642, 15, 2165, 2295, 2273, 13884, 16109, 2049, 17, 1, 2, 2, 2, 2]...

First Element of Validation Dataset:
input_ids: [3, 16031, 4715, 3028, 455, 6011, 456, 28576, 11832, 1012, 17, 1788, 2627, 3056, 13392, 23774, 1786, 3991, 19881, 11, 2163, 17651, 12, 1795, 16, 8286, 1533, 25321, 2649, 1800, 11037, 1731, 26945, 13958, 3228, 3444, 16, 3444, 5206, 1570, 4152, 12100, 1495, 4863, 6324, 3069, 15, 3981, 1542, 17]...
token_type_ids: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]...
attention_mask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]...
labels: [3, 2165, 12020, 1753, 12680, 18397, 15364, 6266, 15, 5905, 10592, 19319, 26208, 2451, 1049, 20199, 2878, 2627, 1572, 7116, 12448, 14836, 17, 5031, 12621, 5995, 8868, 1583, 15, 1891, 1723, 2627, 13392, 1510, 12656, 29357, 1015, 2715, 1495, 3297, 10723, 41, 1637, 6427, 1743, 26208, 2451, 1049, 17, 1]...
decoder_input_ids: [3, 3, 2165, 12020, 1753, 12680, 18397, 15364, 6266, 15, 5905, 10592, 19319, 26208, 2451, 1049, 20199, 2878, 2627, 1572, 7116, 12448, 14836, 17, 5031, 12621, 5995, 8868, 1583, 15, 1891, 1723, 2627, 13392, 1510, 12656, 29357, 1015, 2715, 1495, 3297, 10723, 41, 1637, 6427, 1743, 26208, 2451, 1049, 17]...
```
These preprocessed examples are critical for model training, ensuring that both the training and validation datasets are ready for optimal learning and inference.

## Load Pre-trained Model
The process of loading pre-trained models is a crucial step in setting up the Bert2GPT Indonesian Text Summarizer, which uses the established weights and configurations of BERT and GPT-2. These models serve as the encoder and decoder, respectively, in a sequence-to-sequence architecture.
### 1. Loading the Pre-trained BERT Model
The summarization model uses a specific pre-trained BERT model called "cahya/bert-base-indonesian-1.5G".
BERT acts as the encoder, designed to understand and process the input text. Since this BERT model is pre-trained on Indonesian language data, it's well-suited for comprehending the context and linguistic features of Indonesian texts.
In this step, BERT reads and encodes the input articles, generating contextual embeddings—representations of the text’s meaning and structure, which are used in the summarization process.

<img width="1094" alt="Screen Shot 2024-10-25 at 21 15 49" src="https://github.com/user-attachments/assets/fcf56efc-c5ff-48e8-af6e-0defa4443c84">

### 2. Loading the Pre-trained GPT-2 Model
The second step involves loading the GPT-2 model, specifically "cahya/gpt2-small-indonesian-522M".
GPT-2 is used as the decoder in this architecture. However, to make it work effectively in this summarization task, it needs to be modified to enable cross-attention. This means GPT-2 can now focus on the encoded information from the BERT model.
After receiving the encoded input from BERT, GPT-2 generates the actual summary. It ensures that the output is not only grammatically fluent but also aligned with the meaning and context of the original article.

<img width="1095" alt="Screen Shot 2024-10-25 at 21 15 58" src="https://github.com/user-attachments/assets/28d00587-938f-421b-b2c3-1e539904db73">

### Why This Approach Works
By combining the pre-trained BERT and GPT-2 models, this summarizer leverages their strengths:

1. BERT excels at understanding context and capturing the deep nuances of the Indonesian language.
2. GPT-2 specializes in generating text that is coherent and fluent.

This setup significantly enhances the model's ability to generate accurate and meaningful summaries while saving time and resources, as the pre-trained models reduce the need for building and training the system from scratch.

## Create Model
The process of creating the Bert2GPT Indonesian Text Summarizer involves carefully setting up the architecture that integrates pre-trained BERT and GPT-2 models into an encoder-decoder framework. Here’s an explanation of each step based on the code and process you've outlined:

### 1. Create the GPT-2 Model with Modified Configuration
The GPT-2 model is initialized with a configuration that allows cross-attention. This ensures that the GPT-2 decoder can attend to the output from the BERT encoder.
Cross-attention enables the decoder to focus on relevant parts of the encoded input text, making the summarization process more accurate and context-aware.
### 2. Combine BERT and GPT-2 into an Encoder-Decoder Model
The BERT model is the encoder, and the GPT-2 model is the decoder in this architecture. The encoder reads the input text, converts it into a meaningful representation (encoded data), and then passes this representation to the GPT-2 decoder.
The decoder then generates a summary based on the encoded input. This sequence-to-sequence structure is typical for tasks like summarization.
### 3. Update Special Tokens Based on the Tokenizer
Special tokens such as the decoder start token, pad token, and end-of-sequence (EOS) token are essential for the model to understand where a sequence begins and ends.
These tokens are crucial during training and inference, ensuring the model handles inputs of varying lengths correctly.
### 4. Check for GPU Availability
GPU availability is checked to determine whether the model can be trained on a GPU, which would significantly speed up training.
If a GPU is available, the model is moved to the GPU for faster computation. Otherwise, it will default to the CPU, which might take longer for training but is still functional.
### 5. Move the Model to the Appropriate Device
Once the device (GPU or CPU) is determined, the model is moved to that device to ensure it's ready for training.
### 6. Print Model Architecture and Parameters
The code prints the architecture of the model to give a clear overview of how the components are structured.
Additionally, it calculates the total number of parameters and the trainable parameters in the model. This information is helpful to understand the model's complexity and its trainable components.
### Summary
The steps taken in this process ensure that the Bert2GPT Indonesian Text Summarizer is effectively prepared for training. By integrating pre-trained BERT and GPT-2 models and configuring special tokens, the model leverages advanced language understanding and text generation capabilities. Checking GPU availability helps optimize the training process for faster results, and understanding the model’s parameters gives insight into its complexity.

## Training Model
The training process of the Bert2GPT Indonesian Text Summarizer involves a structured approach, consisting of two key phases: Initial Training and Continued Training. 
### 1. Initial Training
This phase involves setting up the environment and the necessary configurations to train the model from scratch or from a pre-trained state.
#### Key Steps in Initial Training:
##### Training Arguments:
Define how the training process will be managed, such as:
1. Epochs: The model is set to train for 3 epochs, allowing it to iteratively refine its understanding of the data.
2. Batch Sizes: The training and evaluation batches are set to 16 samples, ensuring a balance between performance and memory utilization.
3. Warmup Steps: 500 warmup steps are included, where the learning rate gradually increases, helping stabilize training initially.
4. Weight Decay: A weight decay of 0.01 is implemented as a regularization technique to prevent overfitting.
5. Mixed Precision: The training utilizes mixed precision (FP16) to accelerate computation and reduce memory usage, allowing for larger batch sizes.
6. Logging and Checkpoints: Logging is configured to capture results every 100 steps, while model checkpoints are saved every 1,000 steps to minimize storage usage.
7. Gradient Accumulation: The training process incorporates gradient accumulation over 4 steps, allowing for effective training with limited memory resources.
#### Trainer Creation:
An instance of the `Trainer class` is created to facilitate the training process. The trainer manages data loading, optimizes the model weights, and evaluates performance against a validation dataset. It abstracts much of the complexity involved in training, allowing for a more streamlined approach.
#### Model Training:
The training process is initiated by calling the `train()` method on the trainer instance. During this phase, the model's weights are adjusted iteratively to minimize the loss function, enhancing the model's summarization capabilities. The training output includes updates on the training and validation loss at specified intervals, providing insight into the model's learning progress.
#### Model Saving
Upon completion of the optimized training, the model is saved to a specified directory, ensuring that the trained weights and configurations are preserved for future use.

### 2. Continue Training
The Continued Training phase allows for the refinement of the model after it has been trained for a while. Instead of starting from scratch, training resumes from a saved checkpoint to further enhance performance.
#### Key Steps in Continued Training:
##### Loading the Model from a Checkpoint:
The model and tokenizer are reloaded from a specified checkpoint directory, allowing the training to resume from the last saved state. This ensures that the model retains the knowledge gained during the initial training phase.
##### Defining Continued Training Arguments:
The training arguments for continued training are defined with slight adjustments to optimize the process:
1. Number of Epochs: Set to 2 epochs to allow for further fine-tuning.
2. Batch Sizes: Maintained at 16 samples for both training and evaluation.
3. Learning Rate: Reduced to 2e-5, which is lower than the initial training rate, promoting finer adjustments to the model weights.
4. Warmup Steps: Set to 500 to allow a gradual increase in the learning rate at the start of training.
5. Logging and Checkpoints: Similar logging intervals and checkpointing strategies are retained for consistency in monitoring the training progress.
6. Best Model Loading: Configured to load the best model at the end of training based on evaluation loss.
##### Reinitializing the Trainer:
A new instance of the `Trainer` class is created, utilizing the loaded model and the updated training arguments. This setup allows the training process to continue seamlessly from the previous checkpoint.
##### Continuing Training:
The training process is resumed by calling the `train()` method on the trainer instance. During this phase, the model continues to adjust its weights to minimize the loss function based on the remaining optimization steps. Training output includes updates on training and validation loss at specified intervals, providing insight into the model's improvement.

##### Model Saving
Upon completion of the continued training, the final model is saved to a designated directory, preserving the trained weights and configurations for future use. Additionally, the tokenizer is saved alongside the model to ensure compatibility during inference.

### Summary of Phases
1. Initial Training: Starts the training from the beginning or a pre-trained state, using well-defined training arguments and datasets.
2. Continue Training: Resumes training from a checkpoint, allowing incremental improvements to the model while saving time and resources.
Both phases allow for flexibility, letting the model evolve through initial training and further refinement through continued training. This iterative approach helps achieve better performance over time while utilizing pre-existing model checkpoints.

## Evaluate Model
The model evaluation process consists of two key components: tracking the training progress via loss graphs and measuring the model's performance using ROUGE scores. 

### Training and Validation Losses

![1a](https://github.com/user-attachments/assets/04ca560c-3700-461b-8abc-b4e47235d9b2)

=== Training Summary ===
Total Steps: 4800
Number of Evaluations: 4

Training Loss:
Initial: 0.2962
Final: 0.2593
Minimum: 0.2546
Mean: 0.2743
Improvement: 0.0369

Validation Loss:
Initial: 0.2983
Final: 0.2768
Minimum: 0.2768
Mean: 0.2860
Improvement: 0.0215

The training and validation loss metrics demonstrate the effectiveness of the continued training phase for the Bert2GPT Indonesian Text Summarizer. The improvements in both metrics highlight the model's enhanced capacity for generating accurate summaries. These findings underscore the importance of monitoring loss during training to ensure that the model develops robust summarization capabilities. If further evaluations or optimizations are needed, monitoring these trends will be crucial for guiding the next steps in model refinement.

### ROUGE Scores Evaluation
The evaluation of the model's performance is presented through a comprehensive analysis of ROUGE scores, summary length distributions, and length correlations.

![1b](https://github.com/user-attachments/assets/23c9eb5e-e624-43de-aa22-a9d114c976f1)

#### ROUGE Scores Comparison
The first visualization illustrates the comparison of ROUGE scores across different metrics:
ROUGE-1 Scores:
- Recall: 0.675 (67.5%)
- Precision: 0.236 (23.6%)
- F1: 0.349 (34.9%)

ROUGE-2 Scores:
- Recall: 0.277 (27.7%)
- Precision: 0.071 (7.1%)
- F1: 0.113 (11.3%)

ROUGE-L Scores:
- Recall: 0.594 (59.4%)
- Precision: 0.206 (20.6%)
- F1: 0.306 (30.6%)
These scores reflect the model's ability to capture n-grams and the longest common subsequence between generated and reference summaries.
#### ROUGE ROUGE Scores Heatmap
The heatmap further visualizes the ROUGE scores, providing an at-a-glance understanding of the performance across different metrics. The color intensity indicates the magnitude of each score, with higher values represented in deeper colors.


![1c](https://github.com/user-attachments/assets/2859a19d-cbcd-4d3d-8434-039b40a3d308)

#### Summary Length Distribution
Thee distribution of summary lengths for both generated and reference summaries:
- Generated Summaries: The box plot reveals that generated summaries tend to be longer, with a range of 58 to 69 words.
- Reference Summaries: The reference summaries are shorter, with a range of 12 to 20 words. This highlights that the model generates more extensive summaries compared to the provided references.
#### Length Correlation
The scatter plot on the right illustrates the correlation between the lengths of the reference summaries and the generated summaries. Although there is a positive correlation, it indicates that the generated summaries tend to be significantly longer than the references, suggesting that the model may be providing more

#### Summary Length Analysis
Length Statistics:
                Generated    Reference
Count           5.000000    5.000000
Mean           62.400000   16.600000
Std             4.505552    3.130495
Min            58.000000   12.000000
Median         60.000000   16.000000
Max            69.000000   20.000000

#### Example Analysis
`Example Summary:
Reference (16 words): 
"Tim peneliti ITB kembangkan teknologi pengolahan limbah ramah lingkungan 
yang berpotensi mengurangi polusi air dan udara."

Generated (60 words):
"[UNK] bandung - sebuah tim peneliti dari institut teknologi bandung (itb) 
baru-baru ini berhasil mengembangkan teknologi baru dalam pengolahan 
limbah yang lebih ramah lingkungan. teknologi ini, yang menggunakan 
proses biologis untuk mengurai limbah industri, diharapkan dapat secara 
signifikan mengurangi polusi air dan udara di sekitar area industri."

Length Ratio: 3.75`

#### Performance Metrics Assessment
✓ All key ROUGE metrics meet acceptable thresholds:

ROUGE-1 F1 > 0.3 (Achieved: 0.349)
ROUGE-2 F1 > 0.1 (Achieved: 0.113)
ROUGE-L F1 > 0.25 (Achieved: 0.306)

### Advanced Model Evaluation Metrics

![1d](https://github.com/user-attachments/assets/9eb4c9d8-3c72-432c-9288-6bcc64640c4b)

The evaluation of the summarization model involves a detailed look at several key metrics, providing insights into its performance.

#### Distribution of Scores
The first chart showcases how different evaluation metrics stack up:

- BLEU Score: This metric measures the overlap of n-grams (contiguous sequences of words) between the generated summaries and the reference texts. A lower BLEU score indicates challenges in retaining the original phrasing.
- METEOR Score: Unlike BLEU, METEOR evaluates semantic similarity, taking into account synonyms and stemming, which helps capture a broader range of meanings. The scores here show a moderate variability.
- Coverage: This metric assesses how much of the reference content is included in the generated summary, revealing the model's effectiveness in capturing essential points.
- Sentiment Similarity: This score indicates how closely the sentiment of the generated summaries matches that of the reference summaries.
The box plot reveals that BLEU scores have the least variation and the highest median, suggesting it's the most consistent metric for evaluating summarization quality.

#### Length Ratios Distribution
In this section, we examine the length ratios of the generated summaries compared to their reference counterparts. An ideal length ratio of 1 (meaning the generated summary is the same length as the reference) is marked with a red dashed line. The histogram indicates that most generated summaries tend to be longer than the references, which may suggest the need for better control over summary length.

#### Redundancy vs. Coverage
The scatter plot illustrates the relationship between redundancy scores (how repetitive the generated summaries are) and content coverage (how much reference content is captured). Lower redundancy scores imply a richer vocabulary in the summaries, while higher coverage indicates a closer alignment with reference texts. The spread of the points suggests room for improvement in balancing these two aspects.

#### Per-Summary Performance Heatmap
The heatmap offers a breakdown of performance for each summary across various metrics, including BLEU, METEOR, coverage, and sentiment similarity. This visual aid helps identify which summaries performed well and which need enhancement.

## Results
### Testing the Model
#### 1. Load the Tokenizer and Model
The tokenizer used is "cahya/bert2gpt-indonesian-summarization," which converts text into tokens suitable for model processing. Special tokens are set for the beginning (BOS) and end (EOS) of sequences.
The model is loaded from a checkpoint, enabling it to utilize previously learned parameters for generating summaries.
#### 2. Test Articles with Reference Summaries
Two test articles were provided, each accompanied by a reference summary. These articles cover topics related to the Indonesian military's political neutrality and the PKS party's stance on corruption.
#### 3. Generate Summary
The function generate_summary(article) generates a summary of the input article using parameters like minimum and maximum length, beam search for optimizing output, repetition penalties to avoid redundancy, and various sampling techniques for more diverse output.
#### 4. Evaluate Summary
The function `evaluate_summary(generated, reference)` computes several metrics, including:
- ROUGE scores to assess overlap with the reference summaries.
- BLEU score for n-gram precision.
- Length ratio to compare the length of the generated summary against the reference.
#### 5. Run Tests and Collect Results
Each article is processed, and the generated summaries are compared to their respective references, yielding metrics for evaluation.

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Test Results</title>
    <style>
        body {
            font-family: Arial, sans-serif;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #dddddd;
            text-align: left;
            padding: 8px;
        }
        th {
            background-color: #f2f2f2;
        }
        h2 {
            text-align: center;
        }
    </style>
</head>
<body>

<h2>Model Test Results</h2>

<table>
    <tr>
        <th>Article</th>
        <th>Generated Summary</th>
        <th>Reference Summary</th>
        <th>ROUGE-1 F</th>
        <th>ROUGE-2 F</th>
        <th>ROUGE-L F</th>
        <th>BLEU</th>
        <th>Length Ratio</th>
    </tr>
    <tr>
        <td>Article 1</td>
        <td>[UNK] mantan presiden pks hidayat nur wahid mengatakan, tni adalah alat negara yang harus netral dan berada di atas seluruh kekuatan politik yang ada. ini untuk menjaga keamanan teritorial dan keutuhan negara kesatuan republik indonesia. kata hidayat, anggota komisi i dpr ri itu. akan ditentukan oleh undang - undangnya dalam pemilu atau pilkada jika sudah tidak ada hambatan yang mengganggu kekesalan. apa pun terjadi di dpr?</td>
        <td>TNI harus profesional dan netral dalam politik. Pemberian hak pilih TNI dalam pemilu perlu diatur melalui undang-undang.</td>
        <td>0.1600</td>
        <td>0.0247</td>
        <td>0.1600</td>
        <td>0.0000</td>
        <td>3.8824</td>
    </tr>
    <tr>
        <td>Article 2</td>
        <td>[UNK] pks mengingatkan agar pemerintah tidak tebang pilih dalam menindak para koruptor. pks juga meminta kpk lebih aktif dalam menangani kasus - kasus korupsi besar yang merugikan negara. namun, pks mengimbau agar penegakan hukum dilakukan secara adil dan tak tebangnya. beberapa hari ini ( 18 / 6 ). " kata ketua dpp pks hidayat nur wahid di jakarta, jumat ( 19 / 9 )</td>
        <td>PKS mendukung pemberantasan korupsi dan meminta penegakan hukum dilakukan secara adil tanpa tebang pilih.</td>
        <td>0.2899</td>
        <td>0.1316</td>
        <td>0.2319</td>
        <td>0.0605</td>
        <td>4.5714</td>
    </tr>
</table>

<h2>Average Metrics Across All Tests</h2>
<table>
    <tr>
        <th>Metric</th>
        <th>Average Value</th>
    </tr>
    <tr>
        <td>ROUGE-1 F</td>
        <td>0.2249</td>
    </tr>
    <tr>
        <td>ROUGE-2 F</td>
        <td>0.0781</td>
    </tr>
    <tr>
        <td>ROUGE-L F</td>
        <td>0.1959</td>
    </tr>
    <tr>
        <td>BLEU</td>
        <td>0.0302</td>
    </tr>
    <tr>
        <td>Length Ratio</td>
        <td>4.2269</td>
    </tr>
</table>

<h2>Recommendations</h2>
<ul>
    <li>Content Preservation: Focus on improving how well the model retains key information from the source text.</li>
    <li>Phrase-Level Accuracy: Aim to enhance the accuracy of phrases to achieve better ROUGE-2 scores.</li>
    <li>Length Conciseness: Adjust parameters to ensure summaries are more concise, especially since the average length ratio exceeds 2.</li>
    <li>Beam Search Parameters: Review and tweak beam search parameters to enhance fluency and coherence in generated summaries.</li>
</ul>

</body>
</html>
