import React, { useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";

const labels = ["いくら", "マグロ", "いか", "うに", "たまご", "えび"];

const RealTimeImageRecognition: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [model, setModel] = useState<tf.GraphModel | null>(null);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [useBackCamera, setUseBackCamera] = useState(true);

  useEffect(() => {
    const loadModel = async () => {
      try {
        const loadedModel = await tf.loadGraphModel(
          "mobilenet/tfjs_model/model.json"
        );
        setModel(loadedModel);
        console.log("Model loaded successfully");
      } catch (error) {
        console.error("Failed to load model:", error);
      }
    };

    const fetchDevices = async () => {
      const mediaDevices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = mediaDevices.filter(
        (device) => device.kind === "videoinput"
      );
      setDevices(videoDevices);
    };

    const setupCamera = async (useBack: boolean) => {
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        try {
          const backCamera = devices.find((device) =>
            device.label.toLowerCase().includes("back")
          );
          const frontCamera = devices.find((device) =>
            device.label.toLowerCase().includes("front")
          );

          const selectedDevice = useBack ? backCamera : frontCamera;
          const deviceId = selectedDevice?.deviceId;

          const stream = await navigator.mediaDevices.getUserMedia({
            video: deviceId ? { deviceId: { exact: deviceId } } : true,
          });

          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            videoRef.current.play();
          }
        } catch (error) {
          console.error("Failed to access the camera:", error);
        }
      }
    };

    loadModel();
    fetchDevices();
    setupCamera(useBackCamera); // 外カメラを最初に設定
  }, [useBackCamera, devices]);

  const handleCameraToggle = () => {
    setUseBackCamera((prev) => !prev); // カメラを切り替え
  };

  const detectObjects = async () => {
    if (model && videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const context = canvas.getContext("2d");

      if (context) {
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const imageTensor = tf.browser.fromPixels(canvas).expandDims(0);

        const predictions = (await model.executeAsync(
          imageTensor
        )) as tf.Tensor[];
        imageTensor.dispose();

        const classLabels = (predictions[0].arraySync() as number[][])[0];
        const boxes = (predictions[1].arraySync() as number[][][])[0];
        const scores = predictions[4].dataSync() as Float32Array;

        const threshold = 0.5;
        for (let i = 0; i < scores.length; i++) {
          if (scores[i] > threshold) {
            const [y1, x1, y2, x2] = boxes[i];
            const x = x1 * canvas.width;
            const y = y1 * canvas.height;
            const width = (x2 - x1) * canvas.width;
            const height = (y2 - y1) * canvas.height;

            context.strokeStyle = "red";
            context.lineWidth = 2;
            context.strokeRect(x, y, width, height);

            context.fillStyle = "red";
            context.font = "16px Arial";
            context.fillText(
              `${labels[classLabels[i] - 1]}, Score: ${(
                scores[i] * 100
              ).toFixed(1)}%`,
              x,
              y - 10
            );
          }
        }

        predictions.forEach((tensor) => tensor.dispose());
      }
    }
  };

  useEffect(() => {
    const intervalId = setInterval(detectObjects, 100);
    return () => clearInterval(intervalId);
  }, [model]);

  return (
    <div style={{ textAlign: "center" }}>
      <h1>HコースD班くま寿司の寿司検出！！</h1>
      <button onClick={handleCameraToggle}>
        {useBackCamera ? "内カメラに切り替え" : "外カメラに切り替え"}
      </button>
      <div
        style={{
          position: "relative",
          width: "100%",
          maxWidth: "640px",
          aspectRatio: "4 / 3",
          margin: "0 auto",
        }}
      >
        <video
          ref={videoRef}
          style={{
            width: "100%",
            height: "auto",
            display: "none",
          }}
        />
        <canvas
          ref={canvasRef}
          style={{
            width: "100%",
            height: "auto",
          }}
        />
      </div>
    </div>
  );
};

export default RealTimeImageRecognition;
