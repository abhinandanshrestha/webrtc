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

        positive_responses_1 = ["Yes ", "Yes, Yep", "Yes, I am David ", "Yes I'm David", "Yep", "Yeah, I'm David", "Yeah I'm David Speaking","Yeah, tell me"]
        negative_responses_1 = ["No",  "Nope", "No, I'm Suraj", "Nah", "No I'm not David", "I am not David", "Not David"]
        
        positive_responses_2 = ["yeah im free","Yes","Sure", "Yes Please", "Okay"," Yeah, we can talk","Yes I have a moment","You can say","Yes tell me", "it's right time to talk","yes it's the right time","it's appropriate time","yep"]
        negative_responses_2 = ["not the right time","I'm driving right now","im in a meeting","call me later","not today","its not the right time","not right now","not","I'm in middle of something", "Busy right now","busy","No", "No I don't have time", "nope", "not okay", "later", "nop","I don't have a moment","Don't say"]
        
        positive_responses_3 = ["yeah im home tomorrow","Yes im free tomorrow","Yes","yeah sure", "okay", "ok", "that's fine", "fine","you can send","send them"]
        negative_responses_3 = ["call me later","not tomorrow","no","im busy tomorrow","busy","I have meeting to attend","I'm in middle of something","No", "No I don\'t have time","No I don\'t want more details","i'll talk later", "nope", "not okay", "later", "nop"]
        
        positive_responses_4 = ["Yes","Sure", "Yes Please", "Okay", "Yes as soon as possible","yes thank you"]
        negative_responses_4 = ["call me later","Send me next day","not today", "No", "No I don't have time", "nope", "not okay", "later", "nop"]
        
        positive_responses = [positive_responses_1, positive_responses_2, positive_responses_3, positive_responses_4]
        negative_responses = [negative_responses_1, negative_responses_2, negative_responses_3, negative_responses_4]
    
    
        repeat_responses = [["could you speak louder","i can't hear you","repeat","repeat please","can you speak alittle louder?", "pardon?", "can you repeat?", "could you repeat?"]]

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


def top5avg(similarity_list):
    sorted_numbers = sorted(similarity_list, reverse=True)
    top_5 = sorted_numbers[:5]
    return sum(top_5) / len(top_5)

def find_similarity(input_embedding, positive_embeddings, negative_embeddings, repeat_embeddings, threshold=0.5):

    # Calculate cosine similarity between input string and responses

    positive_similarity_scores = [cosine_similarity(input_embedding, embedding)[0][0] for embedding in positive_embeddings]
    negative_similarity_scores = [cosine_similarity(input_embedding, embedding)[0][0] for embedding in negative_embeddings]
    repeat_similarity_scores = [cosine_similarity(input_embedding, embedding)[0][0] for embedding in repeat_embeddings]
    
    top5avg_positive_similarity = top5avg(positive_similarity_scores)
    top5avg_negative_similarity = top5avg(negative_similarity_scores)
    top5avg_repeat_similarity = top5avg(repeat_similarity_scores)

    print(f"top5avg_positive_similarity:{top5avg_positive_similarity}, top5avg_negative_similarity:{top5avg_negative_similarity}, top5avg_repeat_similarity:{top5avg_repeat_similarity}")

    # Check which category has higher similarity
    if (top5avg_positive_similarity >= top5avg_negative_similarity) and (top5avg_positive_similarity >= top5avg_repeat_similarity):
        return 0
    elif (top5avg_repeat_similarity >= top5avg_positive_similarity) and (top5avg_repeat_similarity >= top5avg_negative_similarity):
        return 1
    else:
        return 2
positive_embeds, negative_embeds, repeat_embeds = generate_response_embeddings()

