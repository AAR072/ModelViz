import * as tf from "@tensorflow/tfjs";
import * as tfModels from "@tensorflow-models/knn-classifier";
import Chart from "chart.js/auto";

/**
 * Abstract class for generating data.
 */
abstract class DataGenerator {
  /**
   * Generates a set of data points.
   * @param pointCount The number of data points to generate.
   * @returns An array of data points with x, y values and associated labels.
   */
  public abstract generateData(pointCount: number): { x: number; y: number; label: number }[];
}

/**
 * Shotgun data generator, producing random points with random labels based on x and y comparison.
 */
class ShotgunDataGenerator extends DataGenerator {
  /**
   * Generates a set of shotgun-style data points.
   * @param pointCount The number of data points to generate.
   * @returns An array of data points with x, y values and associated labels.
   */
  generateData(pointCount: number): { x: number; y: number; label: number }[] {
    return Array.from({ length: pointCount }, () => {
      const x = +(Math.random()).toFixed(2);
      const y = +(Math.random()).toFixed(2);
      return { x, y, label: y > x ? 1 : 0 };
    });
  }
}

/**
 * Threshold data generator, labeling based on a threshold value.
 */
class ThresholdDataGenerator extends DataGenerator {
  /**
   * Generates a set of threshold-style data points.
   * @param pointCount The number of data points to generate.
   * @returns An array of data points with x, y values and associated labels.
   */
  generateData(pointCount: number): { x: number; y: number; label: number }[] {
    return Array.from({ length: pointCount }, () => {
      const x = +(Math.random()).toFixed(2);
      const y = +(Math.random()).toFixed(2);
      return { x, y, label: y > 0.5 ? 1 : 0 };
    });
  }
}

/**
 * Linear data generator, labeling based on a linear function of x.
 */
class LinearDataGenerator extends DataGenerator {
  /**
   * Generates a set of linear-style data points.
   * @param pointCount The number of data points to generate.
   * @returns An array of data points with x, y values and associated labels.
   */
  generateData(pointCount: number): { x: number; y: number; label: number }[] {
    return Array.from({ length: pointCount }, () => {
      const x = +(Math.random()).toFixed(2);
      const y = +(Math.random()).toFixed(2);
      return { x, y, label: x > 0.5 ? 1 : 0 };
    });
  }
}

/**
 * KNNVisualizer is used to visualize the classification results in a scatter plot.
 */
class KNNVisualizer {
  private chart: Chart | null = null;

  /**
   * Creates a visualizer instance with a given canvas context.
   * @param ctx The HTML canvas element used for rendering the chart.
   */
  constructor(private ctx: HTMLCanvasElement) {}

  /**
   * Plots the data points and their predictions on the chart.
   * @param data The data points to be plotted.
   * @param predictions The predicted labels corresponding to the data points.
   */
  public plot(data: { x: number; y: number }[], predictions: number[]): void {
    const datasets = data.map((point, index) => ({
      data: [{ x: point.x, y: point.y }],
      backgroundColor: predictions[index] === 1 ? "blue" : "red",
    }));

    // Destroy existing chart if present
    if (this.chart) {
      this.chart.destroy();
    }

    // Create a new scatter plot
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

/**
 * KNNApp integrates the KNN classifier with a visualizer to classify data and display results.
 */
export class KNNApp {
  private classifier: tfModels.KNNClassifier;
  private visualizer: KNNVisualizer;

  /**
   * Initializes the KNNApp with the given canvas context.
   * @param ctx The HTML canvas element used for rendering the visualizer.
   */
  constructor(ctx: HTMLCanvasElement) {
    this.classifier = tfModels.create();
    this.visualizer = new KNNVisualizer(ctx);
  }

  /**
   * Runs the KNN classification process with data generation, training, and prediction.
   * @param functionType The type of data generator to use ("shotgun", "td", or "lr").
   * @param pointCount The number of data points to generate.
   */
  public async run(functionType: string, pointCount: number): Promise<void> {
    const generator = this.getGenerator(functionType);
    if (!generator) {
      console.error("Invalid function type");
      return;
    }

    // Generate data points
    const data = generator.generateData(pointCount);
    const predictions: number[] = [];

    // Train the classifier with the data
    for (const { x, y, label } of data) {
      const feature = tf.tensor([x, y]);
      this.classifier.addExample(feature, label);
      feature.dispose();
    }

    // Make predictions after training
    for (const { x, y } of data) {
      const feature = tf.tensor([x, y]);
      const prediction = await this.classifier.predictClass(feature);
      predictions.push(Number(prediction.label));
      feature.dispose();
    }

    // Visualize the results
    this.visualizer.plot(
      data.map(({ x, y }) => ({ x, y })),
      predictions
    );
  }

  /**
   * Returns the appropriate data generator based on the function type.
   * @param functionType The type of function ("shotgun", "td", or "lr").
   * @returns The data generator for the specified function type, or null if invalid.
   */
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
