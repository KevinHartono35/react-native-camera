package id.forky.model;

import android.content.Context;
import android.graphics.Bitmap;

import org.tensorflow.lite.Interpreter;

import java.util.Arrays;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import id.forky.Utils;

public final class SpoofingModel extends TFLiteModel<Boolean>{

    public SpoofingModel(Context context) {
        super(context);
    }

    private final static String PATH = "feathernet_wrap.tflite";

    @Override
    public Interpreter loadModel(Context context) {
        return Utils.loadModelFromAssets(context, PATH);
    }

    @Override
    public int getWidth() {
        return  224;
    }

    @Override
    public int getHeight() {
        return 224;
    }

    @Override
    Object getOutputBuffer() {
        return new float[1][1024];
    }

    private final static float[] IMG_MEAN_IN_DATA_SET =
            {0.14300402f, 0.1434545f, 0.14277956f};
    private final static float[] IMG_STD_DEVIATION_IN_DATA_SET =
            {0.10050353f, 0.100842826f, 0.10034215f};

    @Override
    public float[] preProcessPixel(int px) {
        float[] result = new float[3];
        result[0] = ((px >> 16) & 0xFF) / 255.0f;
        result[1] = ((px >>  8) & 0xFF) / 255.0f;
        result[2] = (px & 0xFF) / 255.0f;
        for (int i = 0; i < 3; i++)
            result[i] = (result[i] - IMG_MEAN_IN_DATA_SET[i]) / IMG_STD_DEVIATION_IN_DATA_SET[i];
        return result;
    }

    @Override
    protected ByteBuffer preProcessInput(Bitmap bitmap) {
        int WIDTH = getWidth();
        int HEIGHT = getHeight();
        int[] pixels = new int[WIDTH * HEIGHT];
        bitmap = Bitmap.createScaledBitmap(bitmap, WIDTH, HEIGHT, true);
        bitmap.getPixels(pixels, 0, WIDTH, 0, 0, WIDTH, HEIGHT);

        ByteBuffer input = ByteBuffer
                .allocateDirect(WIDTH * HEIGHT * 3 * Float.SIZE / Byte.SIZE)
                .order(ByteOrder.nativeOrder());

        float[][] channels = new float[3][WIDTH * HEIGHT];
        for (int i = 0; i < pixels.length; i++) {
            float[] rgb = preProcessPixel(pixels[i]);
            for (int j = 0; j < 3; j++)
                channels[j][i] = rgb[j];
        }

        for (float[] channel : channels)
            for (float value : channel)
                input.putFloat(value);

        return input;
    }

    // return true if spoofing
    @Override
    Boolean processOutput(Object output) {
        float[][] result = (float[][])output;
        if (result != null) {
            Utils.log("spoofing: " + Arrays.toString(result[0]));
            return result[0][0] < result[0][1];
        }
        return true;
    }

}
