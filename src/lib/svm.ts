import * as tf from "@tensorflow/tfjs";
import * as tfModels from "@tensorflow-models/knn-classifier";
import Chart from "chart.js/auto";

// Abstract Base Class for Data Generation
abstract class DataGenerator {
  public abstract generateData(pointCount: number): { x: number[]; y: number[]; label: number }[];

  protected createFeature(x: number, y: number): tf.Tensor {
    return tf.tensor([x, y]);
  }
}

// Concrete Implementations for Different Function Types
class ShotgunDataGenerator extends DataGenerator {
  generateData(pointCount: number): { x: number[]; y: number[]; label: number }[] {
    const data = [];
    for (let i = 0; i < pointCount; i++) {
      const x = +(Math.random()).toFixed(2);
      const y = +(Math.random()).toFixed(2);
      const label = y > x ? 1 : 0;
      data.push({ x: [x], y: [y], label });
    }
    return data;
  }
}

class ThresholdDataGenerator extends DataGenerator {
  generateData(pointCount: number): { x: number[]; y: number[]; label: number }[] {
    const data = [];
    for (let i = 0; i < pointCount; i++) {
      const x = +(Math.random()).toFixed(2);
      const y = +(Math.random()).toFixed(2);
      const label = y > 0.5 ? 1 : 0;
      data.push({ x: [x], y: [y], label });
    }
    return data;
  }
}

class LinearDataGenerator extends DataGenerator {
  generateData(pointCount: number): { x: number[]; y: number[]; label: number }[] {
    const data = [];
    for (let i = 0; i < pointCount; i++) {
      const x = +(Math.random()).toFixed(2);
      const y = +(Math.random()).toFixed(2);
      const label = x > 0.5 ? 1 : 0;
      data.push({ x: [x], y: [y], label });
    }
    return data;
  }
}

// Visualization Class
class KNNVisualizer {
  
  constructor(private ctx: HTMLCanvasElement) {

  };

  

  private chart: Chart | null = null;

  public plot(data: { x: number[]; y: number[]; label: number }[]): void {
    const chartData = this.formatDataForChart(data);

    if (!this.ctx) {
      console.error("Canvas element with id 'chart' not found");
      return;
    }

    if (this.chart) {
      this.chart.destroy();
    }

    this.chart = new Chart(this.ctx, {
      type: "scatter",
      data: chartData,
      options: {
        plugins: {
          legend: {
            display: false
          }
        },
        scales: {
          x: {
            type: "linear",
            position: "bottom",
            grid: {
              display: true,
              color: "#363636"
            }
          },
          y: {
            type: "linear",
            position: "left",
            grid: {
              display: true,
              color: "#363636"
            }
          }
        }
      }
    });
  }

  private formatDataForChart(data: { x: number[]; y: number[]; label: number }[]): any {
    const datasets: { data: { x: number; y: number }[]; backgroundColor: string }[] = [];

    data.forEach((point) => {
      const color = point.label === 1 ? "blue" : "red";
      datasets.push({
        data: [{ x: point.x[0], y: point.y[0] }],
        backgroundColor: color
      });
    });

    return {
      datasets
    };
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

  public run(functionType: string, pointCount: number): void {
    const generator = this.getGenerator(functionType);
    if (!generator) {
      console.error("Invalid function type");
      return;
    }

    const data = generator.generateData(pointCount);

    data.forEach((point) => {
      const feature = tf.tensor([...point.x, ...point.y]);
      this.classifier.addExample(feature, point.label);
    });

    this.visualizer.plot(data);
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
