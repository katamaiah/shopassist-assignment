import openai
import ast
import re
import pandas as pd
import json

from openai import OpenAI

from openaicredentials import  get_open_ai_key

print("welcome to laptop shopassist")


client = OpenAI()
client.api_key = ""

def product_map_layer(laptop_description):
  delimiter = "#####"
  lap_spec = {
      "GPU intensity":"(Type of the Graphics Processor)",
      "Display quality":"(Display Type, Screen Resolution, Display Size)",
      "Portability":"(Laptop Weight)",
      "Multitasking":"(RAM Size)",
      "Processing speed":"(CPU Type, Core, Clock Speed)"
  }

  values = {'low','medium','high'}

  prompt=f"""
  You are a Laptop Specifications Classifier whose job is to extract the key features of laptops as per their requirements.
  To analyze each laptop, perform the following steps:
  Step 1: Extract the laptop's primary features from the description {laptop_description}
  Step 2: Store the extracted features in {lap_spec} \
  Step 3: Classify each of the items in {lap_spec}into {values} based on the following rules: \
  {delimiter}
  GPU Intensity: low: < for Integrated graphics or entry-level dedicated graphics like Intel UHD > , \n
  medium: < if Mid-range dedicated graphics like M1, AMD Radeon, Intel Iris > , \n
  high: < High-end dedicated graphics like Nvidia > , \n

  Display Quality: low: < if resolution below Full HD (e.g., 1366x768). > , \n
  medium: < if Full HD resolution (1920x1080) or higher. > , \n
  high: < if High-resolution display (e.g., 4K, Retina) with excellent color accuracy and features like HDR support. > \n

  Portability: low: < if laptop weight is less than 1.51 kg > , \n
  medium: < if laptop weight is between 1.51 kg and 2.51 kg> , \n
  high: <if laptop weight is greater than 2.51 kg> \n

  Multitasking: low: < If RAM size is 8GB, 12GB > , \n
  medium: < if RAM size is 16GB > , \n
  high: < if RAM size is 32GB, 64GB> \n

  Processing Speed: low: < if entry-level processors like Intel Core i3, AMD Ryzen 3> , \n
  medium: < if Mid-range processors like Intel Core i5, AMD Ryzen 5 > , \n
  high: < if High-performance processors like Intel Core i7, AMD Ryzen 7 or higher > \n
  {delimiter}

  {delimiter}
  Here are some input output pairs for few-shot learning:
  input1: "The Dell Inspiron is a versatile laptop that combines powerful performance and affordability. It features an Intel Core i5 processor clocked at 2.4 GHz, ensuring smooth multitasking and efficient computing. With 8GB of RAM and an SSD, it offers quick data access and ample storage capacity. The laptop sports a vibrant 15.6" LCD display with a resolution of 1920x1080, delivering crisp visuals and immersive viewing experience. Weighing just 2.5 kg, it is highly portable, making it ideal for on-the-go usage. Additionally, it boasts an Intel UHD GPU for decent graphical performance and a backlit keyboard for enhanced typing convenience. With a one-year warranty and a battery life of up to 6 hours, the Dell Inspiron is a reliable companion for work or entertainment. All these features are packed at an affordable price of 35,000, making it an excellent choice for budget-conscious users."
  output1: {{'GPU intensity': 'medium','Display quality':'medium','Portability':'medium','Multitasking':'high','Processing speed':'medium'}}

  input2: "The Lenovo ThinkPad X1 Carbon is a sleek and lightweight laptop designed for professionals on the go. It is equipped with an Intel Core i7 processor running at 2.6 GHz, providing strong processing capabilities for multitasking and productivity. With 16GB of RAM and an SSD, it offers fast and efficient performance along with ample storage capacity. The laptop features a 14" IPS display with a resolution of 2560x1440, delivering sharp visuals and accurate colors. It comes with Intel UHD integrated graphics for decent graphical performance. Weighing just 1.13 kg, it is extremely lightweight and highly portable. The laptop features an IR camera for face unlock, providing convenient and secure login options. With a three-year warranty and an impressive battery life of up to 12 hours, the Lenovo ThinkPad X1 Carbon ensures reliability and long-lasting productivity. Priced at 130,000, it offers top-notch performance and portability for professionals."
  output2: {{'GPU intensity': 'medium', 'Display quality': 'high', 'Portability': 'high', 'Multitasking':'high', 'Processing speed':'high'}}

  input3: "The Apple MacBook Pro is a high-end laptop that combines top-tier performance with a stunning display. It is equipped with an Intel Core i9 processor running at 2.9 GHz, providing exceptional processing power for demanding tasks and content creation. With 32GB of RAM and an SSD, it offers seamless multitasking and fast storage access for large projects. The laptop features a 16" Retina display with a resolution of 3072x1920, delivering breathtaking visuals and precise color reproduction. It comes with an AMD Radeon graphics card, ensuring smooth graphics performance for professional applications. Weighing 2.02 kg, it is relatively lightweight for its size. The laptop features a True Tone display, adjusting the color temperature to match the ambient lighting for a more natural viewing experience. With a three-year warranty and a battery life of up to 10 hours, the Apple MacBook Pro offers reliability and endurance for professionals. Priced at 280,000, it caters to users who require uncompromising performance and a superior display for their demanding workloads."
  output3: {{'GPU intensity': 'medium', 'Display quality': 'high', 'Portability': 'medium','Multitasking': 'high', 'Processing speed': 'high'}}
  {delimiter}

  Follow the above instructions step-by-step and output the dictionary {lap_spec} for the following laptop {laptop_description}.
  """

#see that we are using the Completion endpoint and not the Chatcompletion endpoint

  response = client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt=prompt,
    max_tokens = 2000,
    # temperature=0.3,
    # top_p=0.4
    )
  return response.choices[0].message.content

def dialogue_mgmt_system():
    conversation = initialize_conversation()
    introduction = get_chat_model_completions(conversation)
    print(introduction + '\n')
    top_3_laptops = None
    user_input = ''

    while(user_input != "exit"):
        user_input = input("")

        moderation = moderation_check(user_input)
        if moderation == 'Flagged':
            print("Sorry, this message has been flagged. Please restart your conversation.")
            break

        if top_3_laptops is None:
            conversation.append({"role": "user", "content": user_input})

            response_assistant = get_chat_model_completions(conversation)

            moderation = moderation_check(response_assistant)
            if moderation == 'Flagged':
                print("Sorry, this message has been flagged. Please restart your conversation.")
                break

            confirmation = intent_confirmation_layer(response_assistant)

            moderation = moderation_check(confirmation)
            if moderation == 'Flagged':
                print("Sorry, this message has been flagged. Please restart your conversation.")
                break

            if "No" in confirmation:
                conversation.append({"role": "assistant", "content": response_assistant})
                print("\n" + response_assistant + "\n")
                print('\n' + confirmation + '\n')
            else:
                print("\n" + response_assistant + "\n")
                print('\n' + confirmation + '\n')
                response = dictionary_present(response_assistant)

                moderation = moderation_check(response)
                if moderation == 'Flagged':
                    print("Sorry, this message has been flagged. Please restart your conversation.")
                    break

                print('\n' + response + '\n')
                print("Thank you for providing all the information. Kindly wait, while I fetch the products: \n")
                top_3_laptops = compare_laptops_with_user(response)

                validated_reco = recommendation_validation(top_3_laptops)

                if len(validated_reco) == 0:
                    print("Sorry, we do not have laptops that match your requirements. Connecting you to a human expert.")
                    break

                conversation_reco = initialize_conv_reco(validated_reco)
                recommendation = get_chat_model_completions(conversation_reco)

                moderation = moderation_check(recommendation)
                if moderation == 'Flagged':
                    print("Sorry, this message has been flagged. Please restart your conversation.")
                    break

                conversation_reco.append({"role": "user", "content": "This is my user profile" + response})

                conversation_reco.append({"role": "assistant", "content": recommendation})

                print(recommendation + '\n')

        else:
            conversation_reco.append({"role": "user", "content": user_input})

            response_asst_reco = get_chat_model_completions(conversation_reco)

            moderation = moderation_check(response_asst_reco)
            if moderation == 'Flagged':
                print("Sorry, this message has been flagged. Please restart your conversation.")
                break

            print('\n' + response_asst_reco + '\n')
            conversation.append({"role": "assistant", "content": response_asst_reco})

def initialize_conversation():
    '''
    Returns a list [{"role": "system", "content": system_message}]
    '''

    delimiter = "####"
    example_user_req = {'GPU intensity': 'high','Display quality': 'high','Portability': 'low','Multitasking': 'high','Processing speed': 'high','Budget': '150000'}

    system_message = f"""

    You are an intelligent laptop gadget expert and your goal is to find the best laptop for a user.
    You need to ask relevant questions and understand the user profile by analysing the user's responses.
    You final objective is to fill the values for the different keys ('GPU intensity','Display quality','Portability','Multitasking','Processing speed','Budget') in the python dictionary and be confident of the values.
    These key value pairs define the user's profile.
    The python dictionary looks like this {{'GPU intensity': 'values','Display quality': 'values','Portability': 'values','Multitasking': 'values','Processing speed': 'values','Budget': 'values'}}
    The values for all keys, except 'budget', should be 'low', 'medium', or 'high' based on the importance of the corresponding keys, as stated by user.
    The value for 'budget' should be a numerical value extracted from the user's response.
    The values currently in the dictionary are only representative values.

    {delimiter}Here are some instructions around the values for the different keys. If you do not follow this, you'll be heavily penalised.
    - The values for all keys, except 'Budget', should strictly be either 'low', 'medium', or 'high' based on the importance of the corresponding keys, as stated by user.
    - The value for 'budget' should be a numerical value extracted from the user's response.
    - 'Budget' value needs to be greater than or equal to 25000 INR. If the user says less than that, please mention that there are no laptops in that range.
    - Do not randomly assign values to any of the keys. The values need to be inferred from the user's response.
    {delimiter}

    To fill the dictionary, you need to have the following chain of thoughts:
    {delimiter} Thought 1: Ask a question to understand the user's profile and requirements. \n
    If their primary use for the laptop is unclear. Ask another question to comprehend their needs.
    You are trying to fill the values of all the keys ('GPU intensity','Display quality','Portability','Multitasking','Processing speed','Budget') in the python dictionary by understanding the user requirements.
    Identify the keys for which you can fill the values confidently using the understanding. \n
    Remember the instructions around the values for the different keys.
    Answer "Yes" or "No" to indicate if you understand the requirements and have updated the values for the relevant keys. \n
    If yes, proceed to the next step. Otherwise, rephrase the question to capture their profile. \n{delimiter}

    {delimiter}Thought 2: Now, you are trying to fill the values for the rest of the keys which you couldn't in the previous step.
    Remember the instructions around the values for the different keys. Ask questions you might have for all the keys to strengthen your understanding of the user's profile.
    Answer "Yes" or "No" to indicate if you understood all the values for the keys and are confident about the same.
    If yes, move to the next Thought. If no, ask question on the keys whose values you are unsure of. \n
    It is a good practice to ask question with a sound logic as opposed to directly citing the key you want to understand value for.{delimiter}

    {delimiter}Thought 3: Check if you have correctly updated the values for the different keys in the python dictionary.
    If you are not confident about any of the values, ask clarifying questions. {delimiter}

    Follow the above chain of thoughts and only output the final updated python dictionary. \n


    {delimiter} Here is a sample conversation between the user and assistant:
    User: "Hi, I am an editor."
    Assistant: "Great! As an editor, you likely require a laptop that can handle demanding tasks. Hence, the laptop should have high multitasking capability. You would also need a high end display for better visuals and editing. May I know what kind of work do you primarily focus on? Are you more involved in video editing, photo editing, or both? Understanding the specific type of editing work will help me tailor my recommendations accordingly. Let me know if my understanding is correct until now."
    User: "I primarily work with After Effects."
    Assistant: "Thank you for providing that information. Working with After Effects involves working with graphics, animations, and rendering, which will require high GPU. Do you work with high-resolution media files, such as 4K videos or RAW photos? Understanding your file sizes will help determine the storage capacity and processing power needed."
    User: "Yes, sometimes I work with 4K videos as well."
    Assistant: "Thank you for the information. Processing 4K vidoes will require a good processor and high GPU. I think we have already determined earlier that you need a high GPU. To ensure I have a complete understanding of your needs, I have one more question: Are you frequently on the go and require a laptop that is lightweight and easy to carry, or do you primarily work from a stationary location?"
    User: "Yes, sometimes I travel but do not carry my laptop."
    Assistant:"Could you kindly let me know your budget for the laptop? This will help me find options that fit within your price range while meeting the specified requirements."
    User: "my max budget is 1.5lakh inr"
    Assistant: "{example_user_req}"
    {delimiter}

    Start with a short welcome message and encourage the user to share their requirements.
    """
    conversation = [{"role": "system", "content": system_message}]
    return conversation

def get_chat_model_completions(messages):
    response = client.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
        max_tokens = 3000
    )
    return response.choices[0].message["content"]
def moderation_check(user_input):
    response = client.Moderation.create(input=user_input)
    moderation_output = response["results"][0]
    if moderation_output["flagged"] == True:
        return "Flagged"
    else:
        return "Not Flagged"

def intent_confirmation_layer(response_assistant):
    delimiter = "####"
    prompt = f"""
    You are a senior evaluator who has an eye for detail.
    You are provided an input. You need to evaluate if the input has the following keys: 'GPU intensity','Display quality','Portability','Multitasking',' Processing speed','Budget'
    Next you need to evaluate if the keys have the the values filled correctly.
    The values for all keys, except 'budget', should be 'low', 'medium', or 'high' based on the importance as stated by user. The value for the key 'budget' needs to contain a number with currency.
    Output a string 'Yes' if the input contains the dictionary with the values correctly filled for all keys.
    Otherwise out the string 'No'.

    Here is the input: {response_assistant}
    Only output a one-word string - Yes/No.
    """


    confirmation = client.completions.create(
                                    model="gpt-3.5-turbo-instruct",
                                    prompt = prompt,
                                    temperature=0)


    return confirmation.choices[0].message.content

def dictionary_present(response):
    delimiter = "####"
    user_req = {'GPU intensity': 'high','Display quality': 'high','Portability': 'medium','Multitasking': 'high','Processing speed': 'high','Budget': '200000 INR'}
    prompt = f"""You are a python expert. You are provided an input.
            You have to check if there is a python dictionary present in the string.
            It will have the following format {user_req}.
            Your task is to just extract and return only the python dictionary from the input.
            The output should match the format as {user_req}.
            The output should contain the exact keys and values as present in the input.

            Here are some sample input output pairs for better understanding:
            {delimiter}
            input: - GPU intensity: low - Display quality: high - Portability: low - Multitasking: high - Processing speed: medium - Budget: 50,000 INR
            output: {{'GPU intensity': 'low', 'Display quality': 'high', 'Portability': 'low', 'Multitasking': 'high', 'Processing speed': 'medium', 'Budget': '50000'}}

            input: {{'GPU intensity':     'low', 'Display quality':     'high', 'Portability':    'low', 'Multitasking': 'high', 'Processing speed': 'medium', 'Budget': '90,000'}}
            output: {{'GPU intensity': 'low', 'Display quality': 'high', 'Portability': 'low', 'Multitasking': 'high', 'Processing speed': 'medium', 'Budget': '90000'}}

            input: Here is your user profile 'GPU intensity': 'high','Display quality': 'high','Portability': 'medium','Multitasking': 'low','Processing speed': 'high','Budget': '200000 INR'
            output: {{'GPU intensity': 'high','Display quality': 'high','Portability': 'medium','Multitasking': 'high','Processing speed': 'low','Budget': '200000'}}
            {delimiter}

            Here is the input {response}

            """
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens = 2000
        # temperature=0.3,
        # top_p=0.4
    )
    return response.choices[0].message.content

def product_map_layer(laptop_description):
  delimiter = "#####"
  lap_spec = {
      "GPU intensity":"(Type of the Graphics Processor)",
      "Display quality":"(Display Type, Screen Resolution, Display Size)",
      "Portability":"(Laptop Weight)",
      "Multitasking":"(RAM Size)",
      "Processing speed":"(CPU Type, Core, Clock Speed)"
  }

  values = {'low','medium','high'}

  prompt=f"""
  You are a Laptop Specifications Classifier whose job is to extract the key features of laptops as per their requirements.
  To analyze each laptop, perform the following steps:
  Step 1: Extract the laptop's primary features from the description {laptop_description}
  Step 2: Store the extracted features in {lap_spec} \
  Step 3: Classify each of the items in {lap_spec}into {values} based on the following rules: \
  {delimiter}
  GPU Intensity: low: < for Integrated graphics or entry-level dedicated graphics like Intel UHD > , \n
  medium: < if Mid-range dedicated graphics like M1, AMD Radeon, Intel Iris > , \n
  high: < High-end dedicated graphics like Nvidia > , \n

  Display Quality: low: < if resolution below Full HD (e.g., 1366x768). > , \n
  medium: < if Full HD resolution (1920x1080) or higher. > , \n
  high: < if High-resolution display (e.g., 4K, Retina) with excellent color accuracy and features like HDR support. > \n

  Portability: low: < if laptop weight is less than 1.51 kg > , \n
  medium: < if laptop weight is between 1.51 kg and 2.51 kg> , \n
  high: <if laptop weight is greater than 2.51 kg> \n

  Multitasking: low: < If RAM size is 8GB, 12GB > , \n
  medium: < if RAM size is 16GB > , \n
  high: < if RAM size is 32GB, 64GB> \n

  Processing Speed: low: < if entry-level processors like Intel Core i3, AMD Ryzen 3> , \n
  medium: < if Mid-range processors like Intel Core i5, AMD Ryzen 5 > , \n
  high: < if High-performance processors like Intel Core i7, AMD Ryzen 7 or higher > \n
  {delimiter}

  {delimiter}
  Here are some input output pairs for few-shot learning:
  input1: "The Dell Inspiron is a versatile laptop that combines powerful performance and affordability. It features an Intel Core i5 processor clocked at 2.4 GHz, ensuring smooth multitasking and efficient computing. With 8GB of RAM and an SSD, it offers quick data access and ample storage capacity. The laptop sports a vibrant 15.6" LCD display with a resolution of 1920x1080, delivering crisp visuals and immersive viewing experience. Weighing just 2.5 kg, it is highly portable, making it ideal for on-the-go usage. Additionally, it boasts an Intel UHD GPU for decent graphical performance and a backlit keyboard for enhanced typing convenience. With a one-year warranty and a battery life of up to 6 hours, the Dell Inspiron is a reliable companion for work or entertainment. All these features are packed at an affordable price of 35,000, making it an excellent choice for budget-conscious users."
  output1: {{'GPU intensity': 'medium','Display quality':'medium','Portability':'medium','Multitasking':'high','Processing speed':'medium'}}

  input2: "The Lenovo ThinkPad X1 Carbon is a sleek and lightweight laptop designed for professionals on the go. It is equipped with an Intel Core i7 processor running at 2.6 GHz, providing strong processing capabilities for multitasking and productivity. With 16GB of RAM and an SSD, it offers fast and efficient performance along with ample storage capacity. The laptop features a 14" IPS display with a resolution of 2560x1440, delivering sharp visuals and accurate colors. It comes with Intel UHD integrated graphics for decent graphical performance. Weighing just 1.13 kg, it is extremely lightweight and highly portable. The laptop features an IR camera for face unlock, providing convenient and secure login options. With a three-year warranty and an impressive battery life of up to 12 hours, the Lenovo ThinkPad X1 Carbon ensures reliability and long-lasting productivity. Priced at 130,000, it offers top-notch performance and portability for professionals."
  output2: {{'GPU intensity': 'medium', 'Display quality': 'high', 'Portability': 'high', 'Multitasking':'high', 'Processing speed':'high'}}

  input3: "The Apple MacBook Pro is a high-end laptop that combines top-tier performance with a stunning display. It is equipped with an Intel Core i9 processor running at 2.9 GHz, providing exceptional processing power for demanding tasks and content creation. With 32GB of RAM and an SSD, it offers seamless multitasking and fast storage access for large projects. The laptop features a 16" Retina display with a resolution of 3072x1920, delivering breathtaking visuals and precise color reproduction. It comes with an AMD Radeon graphics card, ensuring smooth graphics performance for professional applications. Weighing 2.02 kg, it is relatively lightweight for its size. The laptop features a True Tone display, adjusting the color temperature to match the ambient lighting for a more natural viewing experience. With a three-year warranty and a battery life of up to 10 hours, the Apple MacBook Pro offers reliability and endurance for professionals. Priced at 280,000, it caters to users who require uncompromising performance and a superior display for their demanding workloads."
  output3: {{'GPU intensity': 'medium', 'Display quality': 'high', 'Portability': 'medium','Multitasking': 'high', 'Processing speed': 'high'}}
  {delimiter}

  Follow the above instructions step-by-step and output the dictionary {lap_spec} for the following laptop {laptop_description}.
  """

#see that we are using the Completion endpoint and not the Chatcompletion endpoint

  response = client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt=prompt,
    max_tokens = 2000,
    # temperature=0.3,
    # top_p=0.4
    )
  return response.choices[0].message.content

def extract_dictionary_from_string(string):
    regex_pattern = r"\{[^{}]+\}"

    dictionary_matches = re.findall(regex_pattern, string)

    # Extract the first dictionary match and convert it to lowercase
    if dictionary_matches:
        dictionary_string = dictionary_matches[0]
        dictionary_string = dictionary_string.lower()

        # Convert the dictionary string to a dictionary object using ast.literal_eval()
        dictionary = ast.literal_eval(dictionary_string)
    return dictionary

def compare_laptops_with_user(user_req_string):
    laptop_df= pd.read_csv('updated_laptop.csv')
    user_requirements = extract_dictionary_from_string(user_req_string)
   # budget = int(user_requirements.get('budget', '0').replace(',', '').split()[0])
    #This line retrieves the value associated with the key 'budget' from the user_requirements dictionary.
    #If the key is not found, the default value '0' is used.
    #The value is then processed to remove commas, split it into a list of strings, and take the first element of the list.
    #Finally, the resulting value is converted to an integer and assigned to the variable budget.


    filtered_laptops = laptop_df.copy()
    filtered_laptops['Price'] = filtered_laptops['Price'].str.replace(',','').astype(int)
   # filtered_laptops = filtered_laptops[filtered_laptops['Price'] <= budget].copy()
    #These lines create a copy of the laptop_df DataFrame and assign it to filtered_laptops.
    #They then modify the 'Price' column in filtered_laptops by removing commas and converting the values to integers.
    #Finally, they filter filtered_laptops to include only rows where the 'Price' is less than or equal to the budget.

    mappings = {
        'low': 0,
        'medium': 1,
        'high': 2
    }
    # Create 'Score' column in the DataFrame and initialize to 0
    filtered_laptops['Score'] = 0
    for index, row in filtered_laptops.iterrows():
        user_product_match_str = row['laptop_feature']
        laptop_values = extract_dictionary_from_string(user_product_match_str)
        score = 0

        for key, user_value in user_requirements.items():
            if key.lower() == 'budget':
                continue  # Skip budget comparison
            laptop_value = laptop_values.get(key, None)
            laptop_mapping = mappings.get(laptop_value.lower(), -1)
            user_mapping = mappings.get(user_value.lower(), -1)
            if laptop_mapping >= user_mapping:
                ### If the laptop value is greater than or equal to the user value the score is incremented by 1
                score += 1

        filtered_laptops.loc[index, 'Score'] = score

    # Sort the laptops by score in descending order and return the top 5 products
    top_laptops = filtered_laptops.drop('laptop_feature', axis=1)
    top_laptops = top_laptops.sort_values('Score', ascending=False).head(3)

    return top_laptops.to_json(orient='records')

def recommendation_validation(laptop_recommendation):
    data = json.loads(laptop_recommendation)
    data1 = []
    for i in range(len(data)):
        if data[i]['Score'] > 2:
            data1.append(data[i])

    return data1
def initialize_conv_reco(products):
    system_message = f"""
    You are an intelligent laptop gadget expert and you are tasked with the objective to \
    solve the user queries about any product from the catalogue: {products}.\
    You should keep the user profile in mind while answering the questions.\

    Start with a brief summary of each laptop in the following format, in decreasing order of price of laptops:
    1. <Laptop Name> : <Major specifications of the laptop>, <Price in Rs>
    2. <Laptop Name> : <Major specifications of the laptop>, <Price in Rs>

    """
    conversation = [{"role": "system", "content": system_message }]
    return conversation

##Run this code once to extract product info in the form of a dictionary
laptop_df= pd.read_csv('laptop_data2.csv')

## Create a new column "laptop_feature" that contains the dictionary of the product features
laptop_df['laptop_feature'] = laptop_df['Description'].apply(lambda x: product_map_layer(x))
laptop_df.to_csv("updated_laptop.csv",index=False,header = True)

dialogue_mgmt_system()

