from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
import os

def get_embedding(text):
    url = "http://192.168.88.40:8026/embed"
    payload = {"inputs": text}
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    return  response.json()

def generate_response_embeddings():

        positive_responses_1 = ["हो म बिज्ञान अधिकारी बोल्दै छु ", "हजुर हो ", "हजुर बोल्दै छु ", "के कुरा को लागि हो", "हजुर भन्नुस म सुनिरहेको छु", "ओभाओ भन्नुस् न", "हजार भनोस् न के काम पर्‍यो","हजार भनोस् न"]
        negative_responses_1 = ["हैन",  "हैन नि", "म त अर्कै मान्छे हो", "मेरो नाम त रमेश हो", "रंग नम्बर पर्यो", "रङ नम्बर पर्‍यो", "होइन"]
        
        positive_responses_2 = ["मिल्छ","हजुर भन्नुस न", "अहिले मिल्छ", "हजुर मिल छ"]
        negative_responses_2 = [ "अहिले त मिल्दैन", "मिल्दैन", "भोलि मात्रै मिल्छ", "एकै छिन पछि मात्रै मिल्छ", "अहिले मिल्दैन", "हजुर मिल दैन"]
        
        positive_responses_3 = ["हजुर छैन", "छैन", "भुक्तानी गर्न सक्छु", "कुनै समस्या छैन", "समस्या छैन","समस्या हुदैन"]
        negative_responses_3 = ["छ", "सक्दिन", "यो वर्ष गारो होला जस्तो छ", "समस्या छ", "शक दिन मलाई गाह्रो हुन्छ", "मेरो आम्दानी चै एकदम कम भएको", "मैले सब दिनहरूलाई", "मैले यो पालि लोन तिर्न सक्दिन"]
        
        positive_responses_4 = ["चाहन्छु", "हुन्छ", "कल गर्दा हुन्छ"]
        negative_responses_4 = ["मन छैन", "चाहन्न", "गर्दिन", "अहिले गर्न मन छैन", "चाहाँ दिन म"]
        
        positive_responses = [positive_responses_1, positive_responses_2, positive_responses_3, positive_responses_4]
        negative_responses = [negative_responses_1, negative_responses_2, negative_responses_3, negative_responses_4]
    
    
        repeat_responses = [["मैले बुझिन", "मलाई फेरी भनि दिनुस न", "हजुरले के भन्नु भएको मैले बुझिन", "हजुर के भन्नु भाको?"]]

        positive_file_path = 'Model/positive_embeddings.npy'
        negative_file_path = 'Model/negative_embeddings.npy'
        repeat_file_path = "Model/repeat_embeddings.npy"
        
        if os.path.exists(repeat_file_path):
            loaded_embeds = np.load(repeat_file_path)
            repeat_embeds = loaded_embeds.tolist()
        else:
            repeat_embeds = [[get_embedding(item) for item in group] for group in repeat_responses]
            # np.save(negative_file_path, np.array(negative_embeds))
        
        if os.path.exists(positive_file_path):
            loaded_embeds = np.load(positive_file_path)
            positive_embeds = loaded_embeds.tolist()
        else:
            positive_embeds = [[get_embedding(item) for item in group] for group in positive_responses]
            # np.save(positive_file_path, np.array(positive_embeds))
            
        if os.path.exists(negative_file_path):
            loaded_embeds = np.load(negative_file_path)
            negative_embeds = loaded_embeds.tolist()
        else:
            negative_embeds = [[get_embedding(item) for item in group] for group in negative_responses]
            # np.save(negative_file_path, np.array(negative_embeds))
            
        return positive_embeds, negative_embeds, repeat_embeds

def find_similarity(input_embedding, positive_embeddings, negative_embeddings, repeat_embeddings, threshold=0.5):

    # Calculate cosine similarity between input string and responses


    positive_similarity_scores = [cosine_similarity(input_embedding, embedding) for embedding in positive_embeddings]
    negative_similarity_scores = [cosine_similarity(input_embedding, embedding) for embedding in negative_embeddings]
    repeat_similarity_scores = [cosine_similarity(input_embedding, embedding) for embedding in repeat_embeddings]
    
    print(positive_similarity_scores, negative_similarity_scores)
    # Find maximum similarity score for each category
    max_positive_similarity = max(positive_similarity_scores)
    max_negative_similarity = max(negative_similarity_scores)
    max_repeat_similarity = max(repeat_similarity_scores)

    # Check which category has higher similarity
    if (max_positive_similarity >= max_negative_similarity) and (max_positive_similarity >= max_repeat_similarity):
        return 0
    elif (max_repeat_similarity >= max_positive_similarity) and (max_repeat_similarity >= max_negative_similarity):
        return 1
    else:
        return 2

positive_embeds, negative_embeds, repeat_embeds = generate_response_embeddings()

