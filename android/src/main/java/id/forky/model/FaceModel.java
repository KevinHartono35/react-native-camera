package id.forky.model;

import android.content.Context;

import org.tensorflow.lite.Interpreter;

import java.util.Arrays;

import id.forky.Utils;

public class FaceModel extends TFLiteModel<float[]> {

    public FaceModel(Context context) {
        super(context);
    }

    private final static String PATH = "FaceMobileNet_Float32.tflite";

    public Interpreter loadModel(Context context) {
        return Utils.loadModelFromAssets(context, PATH);
    }

    @Override
    public int getWidth() {
        return 112;
    }

    @Override
    public int getHeight() {
        return 112;
    }

    @Override
    Object getOutputBuffer() {
        return new float[1][192];
    }

    @Override
    public float[] preProcessPixel(int px) {
        return new float[]{
                (((px >> 16) & 0xFF) - 127) / 127f,
                (((px >> 8) & 0xFF) - 127) / 127f,
                ((px & 0xFF) - 127) / 127f,
        };
    }

    @Override
    float[] processOutput(Object output) {
        float[][] result = (float[][]) output;
        if (result != null) {
            Utils.log("face: " + Arrays.toString(result[0]));
            return result[0];
        }
        return null;
    }

}
