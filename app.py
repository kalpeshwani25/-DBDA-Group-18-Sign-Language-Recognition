# Import necessary libraries
import cv2
import mediapipe as mp
import joblib
import numpy as np
import streamlit as st

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# For webcam input:
hands = mp_hands.Hands(
    min_detection_confidence=0.7, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

def data_clean(landmark):
    # Your data cleaning function remains the same
    data = landmark[0]
    
    try:
      data = str(data)

      data = data.strip().split('\n')

      garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']

      without_garbage = []

      for i in data:
          if i not in garbage:
              without_garbage.append(i)

      clean = []

      for i in without_garbage:
          i = i.strip()
          clean.append(i[2:])

      finalClean = [ ]
      for i in range(0, len(clean)):
              if (i+1) % 3 != 0:
                  #clean[i] = float(clean[i]) # Original
                  finalClean.append(float(clean[i]))

      
      return([finalClean])


    except:
      return(np.zeros([1,63], dtype=int)[0])

# Create a Streamlit app
st.title("Sign(Digit) Detection")


frame_container = st.empty()

# Run the app in a loop
while cap.isOpened():
    success, frame = cap.read()

    frame = cv2.flip(frame, 1)

    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(128,), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(128,), thickness=1, circle_radius=1),
            )

        cleaned_landmark = data_clean(results.multi_hand_landmarks)

        if cleaned_landmark:
            clf = joblib.load('pkl_digits.pkl')  # Adjust the path accordingly
            y_pred = clf.predict(cleaned_landmark)
            frame_rgb = cv2.putText(frame_rgb, str(y_pred[0]), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the frame using OpenCV and Streamlit
    frame_container.image(frame_rgb, channels="RGB", use_column_width=True)

# Close resources
hands.close()
cap.release()
cv2.destroyAllWindows()