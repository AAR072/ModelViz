import * as tf from "@tensorflow/tfjs";
import * as tfModels from "@tensorflow-models/knn-classifier";
import Chart from "chart.js/auto";

// Abstract Base Class for Data Generation
abstract class DataGenerator {
  public abstract generateData(pointCount: number): { x: number; y: number; label: number }[];

  protected createFeature(x: number, y: number): tf.Tensor {
    return tf.tensor([x, y]);
  }
}

// Concrete Implementations for Different Function Types
class ShotgunDataGenerator extends DataGenerator {
  generateData(pointCount: number): { x: number; y: number; label: number }[] {
    return Array.from({ length: pointCount }, () => {
      const x = +(Math.random()).toFixed(2);
      const y = +(Math.random()).toFixed(2);
      return { x, y, label: y > x ? 1 : 0 };
    });
  }
}

class ThresholdDataGenerator extends DataGenerator {
  generateData(pointCount: number): { x: number; y: number; label: number }[] {
    return Array.from({ length: pointCount }, () => {
      const x = +(Math.random()).toFixed(2);
      const y = +(Math.random()).toFixed(2);
      return { x, y, label: y > 0.5 ? 1 : 0 };
    });
  }
}

class LinearDataGenerator extends DataGenerator {
  generateData(pointCount: number): { x: number; y: number; label: number }[] {
    return Array.from({ length: pointCount }, () => {
      const x = +(Math.random()).toFixed(2);
      const y = +(Math.random()).toFixed(2);
      return { x, y, label: x > 0.5 ? 1 : 0 };
    });
  }
}

// Visualization Class
class KNNVisualizer {
  private chart: Chart | null = null;

  constructor(private ctx: HTMLCanvasElement) {}

  public plot(data: { x: number; y: number }[], predictions: number[]): void {
    const datasets = data.map((point, index) => ({
      data: [{ x: point.x, y: point.y }],
      backgroundColor: predictions[index] === 1 ? "blue" : "red",
    }));

    if (this.chart) {
      this.chart.destroy();
    }

    this.chart = new Chart(this.ctx, {
      type: "scatter",
      data: { datasets },
      options: {
        plugins: { legend: { display: false } },
        scales: {
          x: { type: "linear", position: "bottom", grid: { color: "#363636" } },
          y: { type: "linear", position: "left", grid: { color: "#363636" } },
        },
      },
    });
  }
}

// Main Application Class
export class KNNApp {
  private classifier: tfModels.KNNClassifier;
  private visualizer: KNNVisualizer;

  constructor(ctx: HTMLCanvasElement) {
    this.classifier = tfModels.create();
    this.visualizer = new KNNVisualizer(ctx);
  }

  public async run(functionType: string, pointCount: number): Promise<void> {
    const generator = this.getGenerator(functionType);
    if (!generator) {
      console.error("Invalid function type");
      return;
    }

    const data = generator.generateData(pointCount);
    const predictions: number[] = [];

    for (const { x, y, label } of data) {
      const feature = tf.tensor([x, y]);
      this.classifier.addExample(feature, label);

      feature.dispose(); // Avoid memory leaks
    }
    for (const { x, y } of data) {
      const feature = tf.tensor([x, y]);

      const prediction = await this.classifier.predictClass(feature); // Await the Promise
      predictions.push(Number(prediction.label)); // Safely extract and convert the label

      feature.dispose(); // Avoid memory leaks
    }

    this.visualizer.plot(
      data.map(({ x, y }) => ({ x, y })),
      predictions
    );
  }

  private getGenerator(functionType: string): DataGenerator | null {
    switch (functionType) {
      case "shotgun":
        return new ShotgunDataGenerator();
      case "td":
        return new ThresholdDataGenerator();
      case "lr":
        return new LinearDataGenerator();
      default:
        return null;
    }
  }
}
