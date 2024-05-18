import requests

def fetch_twitter_data(hashtag):
    url = "https://twitter154.p.rapidapi.com/hashtag/hashtag"
    payload = {
        "hashtag":"#"+hashtag,
        "limit": 20,
        "section": "top",
        "language": "en"
    }
    headers = {
        "content-type": "application/json",
        "X-RapidAPI-Key": "79bae17330msh8cb331fb4c1ca2cp1c7b77jsnf4e61fab3193",
        "X-RapidAPI-Host": "twitter154.p.rapidapi.com"
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status() 
        json_data = response.json()
        if "results" in json_data:
            return [tweet['text'] for tweet in json_data["results"]]
        else:
            print("No tweets found in the response.")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Twitter data: {e}")
        return None

