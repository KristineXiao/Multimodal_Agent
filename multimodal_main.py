"""
Multimodal Boston Guide CrewAI Agent
A two-agent system with personalized introduction and Boston recommendations
Now supports both text and speech interaction!
"""

import os
import warnings
import locale
import sys
from crewai import Agent, Task, Crew, Process
from crewai import LLM
from speech_utils import SpeechManager
import re

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

llm = LLM(model="gpt-4o")

def create_introduction_task(user_choice, agent):
    """Create a self-introduction task based on user preference"""
    base_intro = """
    Introduce yourself as Tong in 3-5 sentences. You are a Harvard M.S. Data Science student originally from Shenzhen, China, 
    who studied in Beijing for college. You love street dance (choreography and K-pop), cooking and tasting food, city walks, 
    traveling, exploring new things, artistic experiences, movies, and caring for plants and animals (especially dogs and birds).
    You bring warmth, curiosity, and creativity into conversations.
    """
    
    if user_choice == "1":
        description = base_intro + """
        Focus on your passion for food, cooking, and trying new restaurants. Emphasize your love for Asian cuisine 
        and exploring diverse flavors. Make it clear why food recommendations would be perfect for you.
        """
        expected_output = "A warm 3-5 sentence introduction emphasizing Tong's food interests and dining preferences."
        
    elif user_choice == "2":
        description = base_intro + """
        Focus on your love for activities and experiences. Emphasize your interests in street dance, K-pop, 
        city walks, movies, artistic experiences, and exploring new things. Show your adventurous spirit.
        """
        expected_output = "A warm 3-5 sentence introduction emphasizing Tong's activity interests and adventurous nature."
        
    elif user_choice == "3":
        description = base_intro + """
        Provide a balanced introduction that highlights both your food interests (cooking, trying restaurants) 
        and your activity interests (dancing, movies, city walks, art). Show your well-rounded personality.
        """
        expected_output = "A warm 3-5 sentence balanced introduction covering both Tong's food and activity interests."

    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
        max_iter=1
    )

def create_boston_guide_task(user_choice, agent, intro_task):
    """Create recommendation task that uses the introduction as context"""
    
    base_requirements = """
    You are Tong. Based on your personal introduction from the previous task, give personalized recommendations 
    that align with YOUR interests and background as a Harvard Data Science student.
    
    Requirements:
    - Reference YOUR introduction when explaining why recommendations fit YOUR personality
    - Format as numbered Markdown lists
    - Each item must include ONE emoji and name in bold
    - Add 1-2 sentences explaining why it's perfect for Tong based on the introduction
    - Focus on Cambridge, Allston, Brighton, Boston proper, Brookline, and Somerville
    - Focus on budget-friendly options for students
    """
    
    if user_choice == "1":
        description = base_requirements + """
        - Recommend EXACTLY 3 different student-friendly restaurants
        - Connect each recommendation to your food interests mentioned in the introduction
        - Stop after exactly 3 recommendations
        """
        expected_output = """A numbered Markdown list with exactly 3 restaurants formatted as:
        1. 🍜 **Restaurant Name** - Brief description connecting to your food interests (1-2 sentences)
        2. 🥢 **Restaurant Name** - Brief description connecting to your food interests (1-2 sentences)  
        3. 🌮 **Restaurant Name** - Brief description connecting to your food interests (1-2 sentences)"""

    elif user_choice == "2":
        description = base_requirements + """
        - Recommend EXACTLY 3 different student-friendly activities
        - Connect each recommendation to your activity interests mentioned in the introduction
        - Stop after exactly 3 recommendations
        """
        expected_output = """A numbered Markdown list with exactly 3 activities formatted as:
        1. 🎨 **Activity Name** - Brief description connecting to your interests (1-2 sentences)
        2. 🏃 **Activity Name** - Brief description connecting to your interests (1-2 sentences)
        3. 🎭 **Activity Name** - Brief description connecting to your interests (1-2 sentences)"""

    elif user_choice == "3":
        description = base_requirements + """
        - Recommend EXACTLY 3 restaurants AND 3 activities
        - Connect each recommendation to your interests mentioned in the introduction
        - Stop after exactly 6 total recommendations
        """
        expected_output = """Two numbered Markdown lists:
        ## Restaurants
        1. 🍜 **Restaurant Name** - Brief description connecting to your interests (1-2 sentences)
        2. 🥢 **Restaurant Name** - Brief description connecting to your interests (1-2 sentences)
        3. 🌮 **Restaurant Name** - Brief description connecting to your interests (1-2 sentences)

        ## Activities  
        1. 🎨 **Activity Name** - Brief description connecting to your interests (1-2 sentences)
        2. 🏃 **Activity Name** - Brief description connecting to your interests (1-2 sentences)
        3. 🎭 **Activity Name** - Brief description connecting to your interests (1-2 sentences)"""

    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
        max_iter=1,
        context=[intro_task]  # Use introduction task as context
    )

def get_interaction_mode():
    """Get user's preferred interaction mode"""
    print("🎛️ Choose your interaction mode:")
    print("1. Text only (traditional)")
    print("2. Voice only (speech input/output)")
    print("3. Mixed mode (voice input, text + voice output)")
    
    while True:
        mode_choice = input("Enter your choice (1, 2, or 3): ").strip()
        if mode_choice in ["1", "2", "3"]:
            return mode_choice
        print("❌ Invalid choice! Please enter 1, 2, or 3.")

def parse_voice_choice(text):
    """Parse voice input to determine user choice"""
    if not text:
        return None
    
    text = text.lower().strip()
    
    # Direct number recognition
    if "1" in text or "one" in text:
        return "1"
    elif "2" in text or "two" in text:
        return "2"
    elif "3" in text or "three" in text:
        return "3"
    
    # Keyword recognition
    if any(word in text for word in ["food", "restaurant", "eat", "dining", "cook"]):
        return "1"
    elif any(word in text for word in ["activity", "activities", "things to do", "fun", "experience"]):
        return "2"
    elif any(word in text for word in ["both", "everything", "all", "food and activities"]):
        return "3"
    
    # Default fallback
    return None

def get_user_choice_multimodal(mode, speech_manager):
    """Get user choice with support for different interaction modes"""
    
    choice_prompt = (
        "\n🌟 What would you like recommendations for?\n"
        "Option 1: Food recommendations\n"
        "Option 2: Things to do (activities)\n"
        "Option 3: Both food and activities\n"
    )
    
    if mode == "1":  # Text only
        print(choice_prompt)
        while True:
            user_choice = input("Your choice (1, 2, or 3): ").strip()
            if user_choice in ["1", "2", "3"]:
                return user_choice
            print("❌ Invalid choice! Please type 1, 2, or 3.")
    
    elif mode == "2":  # Voice only
        if speech_manager:
            # Pure voice mode - speak the options
            
            # Speak the full prompt with options
            full_prompt = (
                "Now, what would you like recommendations for? "
                "You have three options: "
                "Say 'one' or 'food' for restaurant recommendations. "
                "Say 'two' or 'activities' for things to do. "
                "Or say 'three' or 'both' for food and activities together. "
                "What's your choice?"
            )
            speech_manager.text_to_speech(full_prompt, use_gtts=True)
            
            # Now try voice input with the fixed STT
            max_attempts = 3
            for attempt in range(max_attempts):
                print(f"🎤 Listening for your choice... (attempt {attempt + 1}/{max_attempts})")
                voice_text = speech_manager.get_voice_input(
                    "",  # No text prompt - pure voice mode
                    max_duration=10  # Give more time
                )
                
                if voice_text:
                    parsed_choice = parse_voice_choice(voice_text)
                    if parsed_choice:
                        choice_names = {"1": "food", "2": "activities", "3": "both"}
                        print(f"✅ You chose: {choice_names[parsed_choice]}")
                        confirmation = f"Perfect! I'll give you {choice_names[parsed_choice]} recommendations."
                        transition = f"Let me start by introducing myself, and then I'll share some amazing {choice_names[parsed_choice]} suggestions that I think you'll really enjoy!"
                        speech_manager.text_to_speech(confirmation, use_gtts=True)
                        speech_manager.text_to_speech(transition, use_gtts=True)
                        return parsed_choice
                    else:
                        print(f"🤔 I heard: '{voice_text}'")
                        if attempt < max_attempts - 1:
                            retry_msg = "I heard you but didn't understand your choice. Please say 'one' for food, 'two' for activities, or 'three' for both."
                            speech_manager.text_to_speech(retry_msg, use_gtts=True)
                else:
                    print("❌ No speech detected.")
                    if attempt < max_attempts - 1:
                        retry_msg = "I didn't hear you clearly. Please speak a bit louder and say your choice: one, two, or three."
                        speech_manager.text_to_speech(retry_msg, use_gtts=True)
            
            # Final fallback after all voice attempts
            print("\n⚠️ Voice recognition isn't working well. Let me switch to text input for this part.")
            fallback_msg = "I'm having trouble with voice recognition right now. Let me ask you to type your choice instead, and then I'll continue speaking to you."
            speech_manager.text_to_speech(fallback_msg, use_gtts=True)
        else:
            print(choice_prompt)
        
        # Fallback to text input
        print("\nWhat would you like recommendations for?")
        print("1. Food recommendations")  
        print("2. Things to do (activities)")
        print("3. Both food and activities")
        
        while True:
            user_choice = input("Your choice (1, 2, or 3): ").strip()
            if user_choice in ["1", "2", "3"]:
                if speech_manager:
                    choice_names = {"1": "food", "2": "activities", "3": "both"}
                    confirmation = f"Perfect! You chose {choice_names[user_choice]} recommendations. Now I'll continue in voice mode."
                    transition = f"Let me introduce myself first, and then I'll share some excellent {choice_names[user_choice]} suggestions with you!"
                    speech_manager.text_to_speech(confirmation, use_gtts=True)
                    speech_manager.text_to_speech(transition, use_gtts=True)
                return user_choice
            print("❌ Invalid choice! Please type 1, 2, or 3.")
    
    elif mode == "3":  # Mixed mode
        print(choice_prompt)
        if speech_manager:
            # Also speak the prompt in mixed mode
            prompt_text = "What would you like recommendations for? You can type one, two, or three, or say your choice out loud."
            speech_manager.text_to_speech(prompt_text, use_gtts=True)
        
        print("🎤 You can type your choice OR press ENTER to use voice input:")
        
        user_input = input("Your choice (1, 2, 3, or ENTER for voice): ").strip()
        
        if user_input in ["1", "2", "3"]:
            # Text input
            if speech_manager:
                choice_names = {"1": "food", "2": "activities", "3": "both"}
                confirmation = f"Perfect! You chose {choice_names[user_input]} recommendations."
                transition = f"Let me start by introducing myself, and then I'll share some great {choice_names[user_input]} suggestions that I think you'll love!"
                speech_manager.text_to_speech(confirmation, use_gtts=True)
                speech_manager.text_to_speech(transition, use_gtts=True)
            return user_input
        elif user_input == "" and speech_manager:
            # Voice input
            print("🎤 Using voice input...")
            speech_manager.text_to_speech("I'm listening for your choice.", use_gtts=True)
            
            voice_text = speech_manager.get_voice_input("", max_duration=8)
            if voice_text:
                parsed_choice = parse_voice_choice(voice_text)
                if parsed_choice:
                    choice_names = {"1": "food", "2": "activities", "3": "both"}
                    print(f"✅ Voice input: You chose {choice_names[parsed_choice]}")
                    confirmation = f"Great! I heard you chose {choice_names[parsed_choice]} recommendations."
                    transition = f"Let me start by introducing myself, and then I'll share some wonderful {choice_names[parsed_choice]} suggestions that I think will be perfect for you!"
                    speech_manager.text_to_speech(confirmation, use_gtts=True)
                    speech_manager.text_to_speech(transition, use_gtts=True)
                    return parsed_choice
                else:
                    print(f"🤔 I heard: '{voice_text}' - let's use text input instead")
                    speech_manager.text_to_speech("I didn't understand that. Let's try typing instead.", use_gtts=True)
            else:
                print("❌ No voice detected - using text input")
                speech_manager.text_to_speech("I didn't hear anything. Let's try typing.", use_gtts=True)
        
        # Fallback to standard text input
        while True:
            user_choice = input("Your choice (1, 2, or 3): ").strip()
            if user_choice in ["1", "2", "3"]:
                if speech_manager:
                    choice_names = {"1": "food", "2": "activities", "3": "both"}
                    confirmation = f"Got it! You chose {choice_names[user_choice]}."
                    transition = f"Now let me introduce myself, and then I'll share some fantastic {choice_names[user_choice]} recommendations that I think will be perfect for you!"
                    speech_manager.text_to_speech(confirmation, use_gtts=True)
                    speech_manager.text_to_speech(transition, use_gtts=True)
                return user_choice
            print("❌ Invalid choice! Please type 1, 2, or 3.")

def output_multimodal(text, mode, speech_manager):
    """Output text with support for different interaction modes"""
    
    # For voice-only mode, minimize text output
    if mode == "2":
        print("🔊 Speaking...")
    else:
        # For text-only and mixed modes, show full text
        print(text)
    
    # Add speech output for voice and mixed modes
    if mode in ["2", "3"] and speech_manager:
        # Clean the text for better speech output
        clean_text = text
        
        # Remove excessive formatting for speech
        clean_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean_text)  # Remove bold formatting
        clean_text = re.sub(r'\*([^*]+)\*', r'\1', clean_text)      # Remove italic formatting  
        clean_text = re.sub(r'#{1,6}\s*', '', clean_text)           # Remove markdown headers
        clean_text = re.sub(r'=+', '', clean_text)                  # Remove separator lines
        clean_text = re.sub(r'[🎓🌟📍👋🍜🥢🌮🎨🏃🎭🎯]', '', clean_text)  # Remove emojis for cleaner speech
        
        # Split into meaningful chunks for better speech pacing
        # Handle different content types
        if "## " in clean_text:  # Has sections like "## Restaurants"
            sections = clean_text.split("## ")
            for section in sections:
                if section.strip():
                    if not section.startswith("## "):
                        section = "## " + section if len(sections) > 1 else section
                    _speak_section(section.strip(), speech_manager)
        else:
            # Regular content - split by lines and paragraphs
            _speak_section(clean_text, speech_manager)

def _speak_section(text, speech_manager):
    """Helper function to speak a section of text naturally"""
    if not text or not text.strip():
        return
    
    # Split into sentences and meaningful chunks
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    for line in lines:
        if len(line) > 3:  # Skip very short lines
            # Handle numbered lists specially
            if re.match(r'^\d+\.', line):
                # This is a numbered item - speak it as one unit
                speech_manager.text_to_speech(line, use_gtts=True)
            elif line.startswith("## "):
                # Section header
                header = line.replace("## ", "")
                speech_manager.text_to_speech(f"Here are my {header.lower()} recommendations:", use_gtts=True)
            else:
                # Regular sentence
                speech_manager.text_to_speech(line, use_gtts=True)

def main():
    # Simple initial greeting - text only
    print("🎓 Multimodal Boston Guide Agent")
    print("=" * 35)
    
    # Initialize speech manager quietly
    speech_manager = None
    try:
        print("🔄 Initializing speech capabilities...")
        speech_manager = SpeechManager(whisper_model_size="base")
        print("✅ Speech system ready!")
    except Exception as e:
        print(f"⚠️ Speech initialization failed: {e}")
        print("📝 Continuing with text-only mode...")
    
    # Get interaction mode first
    mode = get_interaction_mode()
    mode_names = {
        "1": "Text only",
        "2": "Voice only", 
        "3": "Mixed mode"
    }
    
    print(f"\n🎛️ Using {mode_names[mode]} interaction")
    
    # Now provide personalized welcome based on the chosen mode
    base_welcome = (
        "Welcome to Your Harvard Student Digital Twin! "
        "Hi! I'm Tong, and I'm excited to share a bit about myself and recommend some amazing places and food in Boston for you."
    )
    
    if mode == "1":
        # Text only - show full welcome
        print("\n" + "=" * 65)
        print(base_welcome)
        print("=" * 65)
        
    elif mode == "2" and speech_manager:
        # Voice only - minimal text, speak everything
        print("\n🔊 Voice-only mode active - I'll speak to you now...")
        print("=" * 50)
        
        # Personalized voice-only welcome
        voice_welcome = (
            f"{base_welcome} "
            "I'll be speaking to you throughout our conversation, and I'll listen for your voice responses. "
            "Let's begin our voice conversation!"
        )
        speech_manager.text_to_speech(voice_welcome, use_gtts=True)
        
    elif mode == "3":
        # Mixed mode - show text AND speak
        print("\n" + "=" * 65)
        print(base_welcome)
        print("=" * 65)
        
        if speech_manager:
            # Personalized mixed mode intro
            mixed_welcome = (
                f"{base_welcome} "
                "I'll show you text and speak to you as well. You can type or use voice input throughout our conversation. "
                "Let's get started!"
            )
            speech_manager.text_to_speech(mixed_welcome, use_gtts=True)

    # Get user choice with multimodal support
    user_choice = get_user_choice_multimodal(mode, speech_manager)

    if user_choice not in ["1", "2", "3"]:
        print("❌ Invalid choice! Exiting...")
        return

    # Create Agent 1: Self Introduction Agent
    self_intro_agent = Agent(
        role='Tong - Harvard Data Science Student',
        goal='Provide a warm, personalized introduction as Tong that highlights relevant interests based on user preference',
        backstory="""You are Tong, a Harvard M.S. Data Science student originally from Shenzhen, China, 
        who studied in Beijing for college. You love street dance (choreography and K-pop), cooking and tasting food, 
        city walks, traveling, exploring new things, artistic experiences, movies, and caring for plants and animals 
        (especially dogs and birds). You bring warmth, curiosity, and creativity into conversations, balancing 
        technical strength with personal charm. You adapt your introduction based on what the person is interested in.""",
        verbose=False,
        allow_delegation=False,
        llm=llm
    )

    # Create Agent 2: Boston Guide Agent
    boston_guide_agent = Agent(
        role='Tong - Personal Boston Recommender',
        goal='Provide personalized, current Boston recommendations based on personal introduction context',
        backstory="""You are Tong, a Harvard M.S. Data Science student who is good at creating personalized recommendations 
        based on your unique background, interests, and personality. You excel at connecting personal interests 
        to specific places and experiences in the Boston area, especially for students.""",
        verbose=False,
        allow_delegation=False,
        llm=llm
    )

    # Create tasks
    intro_task = create_introduction_task(user_choice, self_intro_agent)
    recommendation_task = create_boston_guide_task(user_choice, boston_guide_agent, intro_task)

    # Create crew with sequential process
    crew = Crew(
        agents=[self_intro_agent, boston_guide_agent],
        tasks=[intro_task, recommendation_task],
        process=Process.sequential,
        verbose=False
    )

    # Run the crew
    try:
        if mode == "1":
            print("\n👋 Let me introduce myself and find perfect recommendations for you...")
        # For voice modes, go straight to introduction without redundant startup message
        
        # Execute the crew
        result = crew.kickoff()
        
        # Get introduction result
        intro_result = intro_task.output.raw if hasattr(intro_task, 'output') else "Introduction completed"
        
        # Output introduction - handle specially for complete speech
        if mode == "1":
            print(intro_result)
        elif mode in ["2", "3"] and speech_manager:
            # For voice modes, speak introduction as one complete unit
            if mode == "2":
                print("🔊 Speaking introduction...")
            else:
                print(intro_result)
            
            # Clean the introduction text for speech
            clean_intro = intro_result
            clean_intro = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean_intro)  # Remove bold formatting
            clean_intro = re.sub(r'\*([^*]+)\*', r'\1', clean_intro)      # Remove italic formatting  
            clean_intro = re.sub(r'#{1,6}\s*', '', clean_intro)           # Remove markdown headers
            clean_intro = re.sub(r'[🎓🌟📍👋🍜🥢🌮🎨🏃🎭🎯]', '', clean_intro)  # Remove emojis
            clean_intro = clean_intro.strip()
            
            # Speak the entire introduction as one unit
            speech_manager.text_to_speech(clean_intro, use_gtts=True)
            
            # Small delay to ensure introduction fully completes before transition
            import time
            time.sleep(1.0)

        # Add a natural transition
        if mode == "1":
            transition_message = "\nNow that you know me better, here are my personalized Boston recommendations just for you!\n"
            print(transition_message)
        elif mode in ["2", "3"] and speech_manager:
            transition_message = "Now that you know more about me, let me share my personalized Boston recommendations with you!"
            speech_manager.text_to_speech(transition_message, use_gtts=True)
        
        # Output recommendations
        output_multimodal(str(result), mode, speech_manager)

        # Save to file
        recommendation_type = {
            "1": "Food Recommendations",
            "2": "Activity Recommendations", 
            "3": "Food & Activity Recommendations"
        }
        
        with open("personalized_boston_guide.txt", "w", encoding="utf-8") as f:
            f.write(f"Tong's Multimodal Boston Guide - {recommendation_type[user_choice]}\n")
            f.write("="*70 + "\n\n")
            f.write(f"Interaction Mode: {mode_names[mode]}\n\n")

            # Save self introduction
            f.write("👋 Self Introduction\n")
            f.write(intro_result + "\n\n")

            # Save recommendations
            f.write("📍 Recommendations\n")
            f.write(str(result) + "\n")
        
        # Natural closing
        if mode == "1":
            print("\n🌟 I hope you enjoy exploring these places in Boston! Your personalized guide has been saved as 'personalized_boston_guide.txt'.")
        elif mode == "2" and speech_manager:
            closing_message = "I hope you enjoy exploring these places in Boston! Your personalized guide has been saved. Have a wonderful time!"
            speech_manager.text_to_speech(closing_message, use_gtts=True)
        elif mode == "3" and speech_manager:
            print("\n🌟 I hope you enjoy exploring these places in Boston!")
            closing_message = "Your personalized guide has been saved. Have a wonderful time exploring!"
            speech_manager.text_to_speech(closing_message, use_gtts=True)

    except Exception as e:
        if "api_key" in str(e).lower():
            error_msg = "💡 Please set your OPENAI_API_KEY environment variable to use the AI features."
            print(f"\n❌ {error_msg}")
            if mode in ["2", "3"] and speech_manager:
                speech_manager.text_to_speech("Please set your OpenAI API key to use the AI features.", use_gtts=True)
        else:
            print(f"\n❌ Error: {str(e)}")
            if mode in ["2", "3"] and speech_manager:
                speech_manager.text_to_speech("Sorry, there was an unexpected error.", use_gtts=True)
        
    finally:
        # Clean up speech manager
        if speech_manager:
            speech_manager.cleanup()

if __name__ == "__main__":
    main()
