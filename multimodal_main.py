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
        print(choice_prompt)
        if speech_manager:
            # Speak the prompt with working TTS
            prompt_text = "What would you like recommendations for? Say one for food, two for activities, or three for both."
            speech_manager.text_to_speech(prompt_text, use_gtts=False)
            
            # Now try voice input with the fixed STT
            max_attempts = 2
            for attempt in range(max_attempts):
                print(f"\n🎤 Voice input (attempt {attempt + 1}/{max_attempts}):")
                voice_text = speech_manager.get_voice_input(
                    "Say your choice: 'one', 'two', or 'three'",
                    max_duration=8
                )
                
                if voice_text:
                    parsed_choice = parse_voice_choice(voice_text)
                    if parsed_choice:
                        choice_names = {"1": "food", "2": "activities", "3": "both"}
                        print(f"✅ Understood: You chose {choice_names[parsed_choice]}")
                        confirmation = f"Great! I understood you chose {choice_names[parsed_choice]}. Let me get those recommendations!"
                        speech_manager.text_to_speech(confirmation, use_gtts=False)
                        return parsed_choice
                    else:
                        print(f"❓ I heard: '{voice_text}' but couldn't understand your choice.")
                        if attempt < max_attempts - 1:
                            speech_manager.text_to_speech("I didn't understand that. Please try again and say one, two, or three.", use_gtts=False)
                else:
                    print("❌ No speech detected.")
                    if attempt < max_attempts - 1:
                        speech_manager.text_to_speech("I didn't hear anything. Let's try again.", use_gtts=False)
            
            # Fallback to text input after voice attempts fail
            print("\n⚠️ Voice input didn't work. Let's use text input:")
            speech_manager.text_to_speech("Let's try typing your choice instead.", use_gtts=False)
        
        # Fallback to text
        while True:
            user_choice = input("Your choice (1, 2, or 3): ").strip()
            if user_choice in ["1", "2", "3"]:
                if speech_manager:
                    choice_names = {"1": "food", "2": "activities", "3": "both"}
                    confirmation = f"Perfect! You chose {choice_names[user_choice]}."
                    speech_manager.text_to_speech(confirmation, use_gtts=False)
                return user_choice
            print("❌ Invalid choice! Please type 1, 2, or 3.")
    
    elif mode == "3":  # Mixed mode
        print(choice_prompt)
        print("🎤 Type your choice (1, 2, or 3) and I'll respond with both text and voice!")
        
        while True:
            user_choice = input("Your choice (1, 2, or 3): ").strip()
            if user_choice in ["1", "2", "3"]:
                if speech_manager:
                    choice_names = {"1": "food", "2": "activities", "3": "both"}
                    confirmation = f"Perfect! You chose {choice_names[user_choice]}."
                    speech_manager.text_to_speech(confirmation, use_gtts=False)
                return user_choice
            print("❌ Invalid choice! Please type 1, 2, or 3.")

def output_multimodal(text, mode, speech_manager):
    """Output text with support for different interaction modes"""
    
    # Always print to console
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
        
        # Split into smaller chunks for better speech pacing
        sentences = [s.strip() for s in clean_text.split('\n') if s.strip()]
        
        for sentence in sentences:
            if sentence and len(sentence) > 3:  # Skip very short lines
                speech_manager.text_to_speech(sentence, use_gtts=True)

def main():
    print("🎓 Welcome to Your Multimodal Harvard Student Digital Twin!")
    print("=" * 65)
    print("Hi! I'm Tong. Let me share a bit about myself and recommend some fun places in Boston.")
    print("This system now supports both text and speech interaction!")
    print("=" * 65)

    # Initialize speech manager
    speech_manager = None
    try:
        print("\n🔄 Initializing speech capabilities...")
        speech_manager = SpeechManager(whisper_model_size="base")
        print("✅ Speech system ready!")
    except Exception as e:
        print(f"⚠️ Speech initialization failed: {e}")
        print("📝 Continuing with text-only mode...")
    
    # Get interaction mode
    mode = get_interaction_mode()
    mode_names = {
        "1": "Text only",
        "2": "Voice only", 
        "3": "Mixed mode"
    }
    
    print(f"\n🎛️ Using {mode_names[mode]} interaction")
    
    if mode in ["2", "3"] and speech_manager:
        speech_manager.text_to_speech(f"Great! We're using {mode_names[mode]} interaction. Let's get started!")

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
        based on someone's unique background, interests, and personality. You excel at connecting personal interests 
        to specific places and experiences in the Boston area, especially for students. You always reference the 
        person's introduction to explain why each recommendation is perfect for them.""",
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
        verbose=True
    )

    # Run the crew
    try:
        startup_message = "\n👋 Let me introduce myself and find perfect recommendations for you..."
        output_multimodal(startup_message, mode, speech_manager)
        
        # Execute the crew
        result = crew.kickoff()
        
        # Get introduction result
        intro_result = intro_task.output.raw if hasattr(intro_task, 'output') else "Introduction completed"
        
        # Output introduction
        output_multimodal(intro_result, mode, speech_manager)

        # Add a transition
        transition_message = "\nNow that you know me better, here are my personalized Boston recommendations just for you!\n"
        output_multimodal(transition_message, mode, speech_manager)
        
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
        
        closing_message = "\n🌟 I hope you like my recommendations and have a great time in Boston!"
        output_multimodal(closing_message, mode, speech_manager)

    except Exception as e:
        error_message = f"\n❌ Error running AI agents: {str(e)}"
        print(error_message)
        if mode in ["2", "3"] and speech_manager:
            speech_manager.text_to_speech("Sorry, there was an error running the AI agents.")
        
        print("💡 Make sure your OPENAI_API_KEY is set correctly.")
        
    finally:
        # Clean up speech manager
        if speech_manager:
            speech_manager.cleanup()

if __name__ == "__main__":
    main()