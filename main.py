import cv2
import numpy as np
import probability_detector
from ultralytics import YOLO
import datetime

# Global variables
text = "JOGADOR 1, MOSTRE SUAS CARTAS"
text_cards = "CARTAS DETECTADAS:"
text_p1 = "JOGADOR 1: (0%)"
text_p2 = "JOGADOR 2: (0%)"
text_t = "MESA: "
text_color = (255, 255, 255)  # Default text color (white)
control_ = 0
mode_ = 0
calling_prob_ = False
cards_p1 = []
cards_p2 = []
cards_t = []
green_card = []
model = YOLO("yolo_weights/best.pt")

game = probability_detector.HoldemTable(num_players=2, deck_type='full')
print(game.view_deck())

# Callback function for mouse events
def on_mouse(event, x, y, flags, param):
    global text_color
    global text
    if event == cv2.EVENT_LBUTTONDOWN:
        set_mode()

# "http://192.168.1.101:8080/video"
cap = cv2.VideoCapture("http://192.168.1.101:8080/video")

# Create a window
window_name = 'JANELINHA DO AMOR'
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, on_mouse)

# Get Winner
def get_winner():
    return "FLA MEN GO"

def convert_card(card):
    card_values = {
        '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8',
        '9': '9', '10': 'T', 'J': 'J', 'Q': 'Q', 'K': 'K', 'A': 'A'
    }
    
    value = card[:-1]
    suit = card[-1].lower()
    
    converted_value = card_values.get(value, value)
    converted_card = converted_value + suit
    
    return converted_card

# Call Model
def call_model(frame):
    global text_cards, green_card
    results = model.predict(frame)
    cards = [results[0].names[int(i)] for i in results[0].boxes.cls]
    cards = list(set(cards))
    set_text_cards(cards)
    green_card = cards.copy()

# Set detected cards to text
def set_text_cards(cards):
    global text_cards
    aux = ""
    for card in cards:
        aux += card + " " 
    text_cards = f"{aux}"

def set_text_player(player, cards, prob=0.0):
    aux = ""
    for card in cards:
        aux += card + " " 
    return f"JOGADOR {player}: {aux} ({prob}%)"

def set_text_table(cards, prob=0.0):
    aux = ""
    for card in cards:
        aux += card + " " 
    return f"MESA: {aux} ({prob}%)"

def set_probs():
    global game, text_p1, text_p2, text_t, cards_p1, cards_p2, cards_t
    probs = game.simulate(num_scenarios=500, odds_type='win_any')
    text_p1 = set_text_player(1, cards_p1, probs['Player 1'])
    text_p2 = set_text_player(2, cards_p2, probs['Player 2'])
    text_t = set_text_table(cards_t, probs['Tie'])

# Call Actual Mode
def set_mode():
    global game, text, mode_, cards_p1, cards_p2, cards_t, text_p1, text_p2, text_t, green_card

    mode_ += 1
    
    if mode_ ==  1:
        text = "Lendo ..."
    elif mode_ == 2:
        text = "JOGADOR 2, MOSTRE SUAS CARTAS"
        cards_ = [convert_card(card) for card in green_card]
        cards_p1 = green_card.copy()
        text_p1 = set_text_player(1, cards_p1)
        game.add_to_hand(1, cards_)
    elif mode_ == 3:
        text = "Lendo ..."
    elif mode_ == 4:
        text = "PRE-FLOP"
        cards_ = [convert_card(card) for card in green_card]
        cards_p2 = green_card.copy()
        game.add_to_hand(2, cards_)
        set_probs()
    elif mode_ == 5:
        text = "Lendo ..."
    elif mode_ == 6:
        text = "FLOP"
        cards_ = [convert_card(card) for card in green_card]
        for card in green_card: cards_t.append(card)
        game.add_to_community(cards_)
        set_probs()
    elif mode_ == 7:
        text = "Lendo ..."
    elif mode_ == 8:
        text = "TURN"
        cards_ = [convert_card(card) for card in green_card]
        for card in green_card: cards_t.append(card)
        game.add_to_community(cards_)
        set_probs()
    elif mode_ == 9:
        text = "Lendo ..."
    elif mode_ == 10:
        text = "RIVER"
        cards_ = [convert_card(card) for card in green_card]
        for card in green_card: cards_t.append(card)
        game.add_to_community(cards_)
        set_probs()
    elif mode_ == 11:
        text = get_winner()

while True:
    ret, frame = cap.read()

    if not ret:
        break
    
    if control_%60 == 0 and mode_%2 == 1:
        cards = call_model(frame)

    # Draw the text with the current color on the frame
    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
    cv2.putText(frame, text_cards, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
    cv2.putText(frame, text_p1, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
    cv2.putText(frame, text_p2, (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
    cv2.putText(frame, text_t, (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
    
    # Show the frame
    cv2.imshow(window_name, frame)
    
    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    control_ += 1

# Release resources
cap.release()
cv2.destroyAllWindows()