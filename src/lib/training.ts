import * as tf from "@tensorflow/tfjs";
import Chart from "chart.js/auto";
export class DataGenerator {
  // static because always only called once
  static createData(functionType: string, maxX: number, minX: number, pointCount: number): [tf.Tensor, tf.Tensor] {
    const xValues: number[] = [];
    const step: number = (maxX - minX) / (pointCount - 1);

    for (let i: number = 0; i < pointCount; i++) {
      xValues.push(minX + i * step);
    }

    let yValues: number[];
    // switch case for readability
    switch (functionType) {
      case "sine":
        yValues = xValues.map((x) => Math.sin(x));
        break;
      case "exponential":
        yValues = xValues.map((x) => Math.pow(2, x));
        break;
      case "parabola":
        yValues = xValues.map((x) => x ** 2);
        break;
      default:
        throw new Error("Invalid function type");
    }

    return [tf.tensor(xValues), tf.tensor(yValues)];
  }
}

export class ModelTrainer {
  model: tf.Sequential;
  private _predictions: number[][];
  framerate: number;

  constructor(model: tf.Sequential, framerate: number) {
    this.model = model;
    this._predictions = [];
    this.framerate = framerate;
  }

  public get predictions(): number[][] {
    return this._predictions;
  }

  compileModel(): void {
    this.model.compile({
      optimizer: tf.train.adam(),
      loss: "meanSquaredError",
      metrics: ["mse"]
    });
  }

  async train(
    functionType: string,
    maxX: number,
    minX: number,
    pointCount: number,
    epochs: number,
    batchSize: number,
    updateCallback: (epoch: number, logs: tf.Logs) => void
  ): Promise<void> {
    const [xData, yData] = DataGenerator.createData(functionType, maxX, minX, pointCount);

    await this.model.fit(xData, yData, {
      epochs: epochs,
      batchSize: batchSize,
      shuffle: true,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          if ((epoch + 1) % this.framerate === 0) {
            const predictionsAtEpoch: number[] = [];
            xData.dataSync().forEach((inputValue) => {
              const prediction: tf.Tensor = this.model.predict(tf.tensor([inputValue])) as tf.Tensor;
              prediction.dataSync().forEach((predictedValue) => predictionsAtEpoch.push(predictedValue));
            });
            this._predictions.push(predictionsAtEpoch);
          }
          updateCallback(epoch, logs);
        },
      },
    });
  }
}

export class Visualization {
  chart: Chart | null = null;

  destroyChart(): void {
    if (this.chart) {
      this.chart.destroy();
    }
  }

  plot(
    xValues: number[],
    yValues: number[],
    predictions: number[][],
    framerate: number
  ): void {
    const ctx: HTMLCanvasElement = document.getElementById("myChart") as HTMLCanvasElement;

    const datasets: { label: string; data: number[]; borderColor: string; backgroundColor: string; fill: boolean; tension: number; }[] = [
      {
        label: "Training Data (Base Function)",
        data: yValues,
        borderColor: "rgb(75, 192, 192)",
        backgroundColor: "rgba(75, 192, 192, 0.2)",
        fill: false,
        tension: 0.1,
      },
    ];

    const colors: string[] = ["red", "orange", "yellow", "green", "purple", "pink", "white"];
    predictions.forEach((predictionSet, index) => {
      datasets.push({
        label: `Epoch ${framerate * (index + 1)} Predictions`,
        data: predictionSet,
        borderColor: colors[index % colors.length],
        backgroundColor: colors[index % colors.length],
        fill: false,
        tension: 0.1
      });
    });

    this.destroyChart();
    this.chart = new Chart(ctx, {
      type: "line",
      data: { labels: xValues, datasets },
      options: {
        responsive: true,
        plugins: {
          title: {
            display: true,
            text: "Model Training and Predictions",
          },
        },
        scales: {
          x: { title: { display: true, text: "X Values" } },
          y: { title: { display: true, text: "Y Values" } },
        },
      },
    });
  }
}
