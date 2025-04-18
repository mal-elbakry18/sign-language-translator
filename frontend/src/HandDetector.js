import React, { useRef, useEffect, useState } from "react";
import Webcam from "react-webcam";
import * as handpose from "@tensorflow-models/handpose";
import * as fp from "fingerpose";
import axios from "axios";

const HandDetector = () => {
  const webcamRef = useRef(null);
  const [landmarkBuffer, setLandmarkBuffer] = useState([]);
  const [prediction, setPrediction] = useState("");

  const sendToBackend = async (frames) => {
    try {
      const res = await axios.post("http://127.0.0.1:5000/predict", {
        landmarks: frames,
      });
      setPrediction(res.data.sentence || res.data.word || res.data.message);
    } catch (err) {
      console.error("Backend error:", err);
    }
  };

  const detectHands = async () => {
    const model = await handpose.load();

    setInterval(async () => {
      if (
        webcamRef.current &&
        webcamRef.current.video.readyState === 4
      ) {
        const video = webcamRef.current.video;
        const predictions = await model.estimateHands(video);

        if (predictions.length > 0) {
          const landmarks = predictions[0].landmarks.flat(); // 21x3 (x,y,z)

          // use x, y only → slice to 42
          const xyLandmarks = landmarks.flatMap((point) => [point[0], point[1]]);
          setLandmarkBuffer((prev) => {
            const updated = [...prev, xyLandmarks].slice(-30);
            if (updated.length === 30) sendToBackend(updated);
            return updated;
          });
        }
      }
    }, 100);
  };

  useEffect(() => {
    detectHands();
  }, []);

  return (
    <div>
      <Webcam
        ref={webcamRef}
        mirrored
        style={{ width: 640, height: 480 }}
      />
      <h3>Prediction: {prediction}</h3>
    </div>
  );
};

export default HandDetector;
