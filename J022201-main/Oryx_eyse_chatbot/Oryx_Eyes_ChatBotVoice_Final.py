import json
import pyttsx3
import speech_recognition as sr
from difflib import get_close_matches
import pygame
import numpy as np
import threading
import queue
import time
from gtts import gTTS
import os
import sys
import math
import cv2
import face_recognition

def load_knowledge_base(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        return json.load(file)

def speak_and_generate_wave(text: str, audio_queue: queue.Queue):
    tts = gTTS(text=text, lang='en')
    temp_file = "output.mp3"
    tts.save(temp_file)

    pygame.mixer.init()
    pygame.mixer.music.load(temp_file)
    pygame.mixer.music.play()

    freq = 440
    while pygame.mixer.music.get_busy():
        wave_data = np.sin(2 * np.pi * np.linspace(0, 1, 1024) * freq)
        audio_queue.put(wave_data)
        time.sleep(0.02)

    pygame.mixer.music.unload()
    os.remove(temp_file)
    audio_queue.put(None)

def visualize_wave(audio_queue: queue.Queue, wave_surface):
    width, height = wave_surface.get_size()
    black = (0, 0, 0)
    blue = (0, 128, 255)
    gray = (128, 128, 128)

    scaling_factor = 50
    fps = 60
    clock = pygame.time.Clock()

    default_wave = np.sin(np.linspace(0, 2 * np.pi, 1024)) * 0.1

    running = True
    while running:
        wave_surface.fill(black)

        if not audio_queue.empty():
            wave_data = audio_queue.get()
            if wave_data is None:
                wave_data = default_wave
        else:
            wave_data = default_wave

        center_y = height // 2

        if np.array_equal(wave_data, default_wave):
            pygame.draw.line(wave_surface, gray, (0, center_y), (width, center_y), 2)
        else:
            data = (wave_data * scaling_factor).astype(int)
            for x in range(len(data) - 1):
                x1 = int(x * width / len(data))
                x2 = int((x + 1) * width / len(data))
                y1 = center_y + data[x]
                y2 = center_y + data[x + 1]
                pygame.draw.line(wave_surface, blue, (x1, y1), (x2, y2), 2)

        clock.tick(fps)

def chat_bot(audio_queue: queue.Queue):
    knowledge_base = load_knowledge_base('index.json')
    first_time = True

    while True:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            if first_time:
                print("I'm listening, please speak now.")
                first_time = False

            try:
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
                user_input = recognizer.recognize_google(audio)
                print(f"You said: {user_input}")

                if user_input.lower() in ["exit", "quit", "goodbye"]:
                    print("Bot: Goodbye!")
                    speak_and_generate_wave("Goodbye!", audio_queue)
                    audio_queue.put(None)
                    break

                best_match = find_best_match(
                    user_input, [q["question"] for q in knowledge_base["questions"]]
                )
                if best_match:
                    answer = get_answer_for_question(best_match, knowledge_base)
                    print(f"Bot: {answer}")
                    speak_and_generate_wave(answer, audio_queue)
                else:
                    print("Bot: I don't know the answer. Can you teach me?")

            except sr.UnknownValueError:
                print("Sorry, I didn't catch that.")
            except sr.RequestError:
                print("Speech recognition service is unavailable.")
            except sr.WaitTimeoutError:
                print("No speech detected. Please try again.")

def find_best_match(user_question: str, questions: list[list[str]]):
    for question_variants in questions:
        match = get_close_matches(user_question, question_variants, n=1, cutoff=0.6)
        if match:
            return match[0]
    return None

def get_answer_for_question(question: str, knowledge_base: dict):
    for q in knowledge_base["questions"]:
        if question in q["question"]:
            return q["answer"]

def eye_movement(screen, eye_surface, cap, known_encodings, labels):
    pupil_happy_left = pygame.image.load(r"Photo\pupil_happy_left.png")
    pupil_happy_right = pygame.image.load(r"Photo\pupil_happy_right.png")
    pupil_angry_left = pygame.image.load(r"Photo\pupil_angry_left.png")
    pupil_angry_right = pygame.image.load(r"Photo\pupil_angry_right.png")
    pupil_normal_left = pygame.image.load(r"Photo\pupil_normal_left.png")
    pupil_normal_right = pygame.image.load(r"Photo\pupil_normal_right.png")

    pupil_happy_left = pygame.transform.scale(pupil_happy_left, (300, 300))
    pupil_happy_right = pygame.transform.scale(pupil_happy_right, (300, 300))
    pupil_angry_left = pygame.transform.scale(pupil_angry_left, (300, 300))
    pupil_angry_right = pygame.transform.scale(pupil_angry_right, (300, 300))
    pupil_normal_left = pygame.transform.scale(pupil_normal_left, (300, 300))
    pupil_normal_right = pygame.transform.scale(pupil_normal_right, (300, 300))

    eye_background = pygame.image.load(r"Photo\eye_background.png")
    eye_background = pygame.transform.scale(eye_background, (600, 600))
    background_image = pygame.image.load(r"Photo\background_image.png")
    background_image = pygame.transform.scale(background_image, (1920, 1080))

    left_eye_center = (960, 220)  
    right_eye_center = (1300, 220)  

    clock = pygame.time.Clock()
    current_pupil_left = pupil_normal_left
    current_pupil_right = pupil_normal_right

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                cap.release()
                cv2.destroyAllWindows()
                sys.exit()

        ret, frame = cap.read()
        rgb_frame = frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        emotion_detected = "طبيعي"
        for face_encoding in face_encodings:
            results = face_recognition.compare_faces(known_encodings, face_encoding)

            if True in results:
                matched_index = results.index(True)
                emotion_detected = labels[matched_index]
                break

        if emotion_detected == "سعيد":
            current_pupil_left = pupil_happy_left
            current_pupil_right = pupil_happy_right
        elif emotion_detected == "غاضب":
            current_pupil_left = pupil_angry_left
            current_pupil_right = pupil_angry_right
        else:
            current_pupil_left = pupil_normal_left
            current_pupil_right = pupil_normal_right

        if face_locations:
            top, right, bottom, left = face_locations[0]
            face_center_x = (left + right) // 2
            face_center_y = (top + bottom) // 2

            target_x = int(face_center_x * 1920 / frame.shape[1])
            target_y = int(face_center_y * 1080 / frame.shape[0])
        else:
            target_x, target_y = 1020, 540

        dx = target_x - 1300
        dy = target_y - 500

        eye_surface.blit(background_image, (0, 0))
        eye_surface.blit(eye_background, (left_eye_center[0] - 300, left_eye_center[1] - 300))
        eye_surface.blit(eye_background, (right_eye_center[0] - 300, right_eye_center[1] - 300))

        eye_surface.blit(current_pupil_left, (int(left_eye_center[0] + dx - 60), int(left_eye_center[1] + dy - 60)))
        eye_surface.blit(current_pupil_right, (int(right_eye_center[0] + dx - 60), int(right_eye_center[1] + dy - 60)))

        pygame.display.flip()
        clock.tick(60)
        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def main():
    pygame.init()
    screen = pygame.display.set_mode((1920, 1080))
    pygame.display.set_caption("Combined Window")

    wave_surface = pygame.Surface((1920, 300))  
    eye_surface = pygame.Surface((1920, 1080))

    cap = cv2.VideoCapture(0)
    known_encodings = []
    labels = []

    happy_image1 = face_recognition.load_image_file(r"Photo\amjed.jpg")
    happy_image2 = face_recognition.load_image_file(r"Photo\ziad.jpg")
    angry_image2 = face_recognition.load_image_file(r"Photo\boda.jpg")

    happy_encoding1 = face_recognition.face_encodings(happy_image1)[0]
    happy_encoding2 = face_recognition.face_encodings(happy_image2)[0]
    angry_encoding2 = face_recognition.face_encodings(angry_image2)[0]

    known_encodings = [happy_encoding1, happy_encoding2, angry_encoding2]
    labels = ["سعيد", "سعيد", "غاضب"]

    audio_queue = queue.Queue()

    t1 = threading.Thread(target=chat_bot, args=(audio_queue,))
    t2 = threading.Thread(target=visualize_wave, args=(audio_queue, wave_surface))
    t3 = threading.Thread(target=eye_movement, args=(screen, eye_surface, cap, known_encodings, labels))

    t1.start()
    t2.start()
    t3.start()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:  
                    running = False

        screen.fill((0, 0, 0))
        screen.blit(eye_surface, (0, 0))  
        screen.blit(wave_surface, (0, 780))  
        pygame.display.flip()

    pygame.quit()
    cap.release()
    cv2.destroyAllWindows()
    sys.exit()

if __name__ == "__main__":
    main()
