package id.forky.model;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.SystemClock;

import org.tensorflow.lite.Interpreter;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import id.forky.Utils;

public abstract class TFLiteModel<T> {

    private Interpreter model;
    private final int WIDTH;
    private final int HEIGHT;

    protected abstract Interpreter loadModel(Context context);

    public boolean isReady() {
        return model != null;
    }

    public abstract int getWidth();

    public abstract int getHeight();

    abstract Object getOutputBuffer();

    abstract float[] preProcessPixel(int px);

    abstract T processOutput(Object output);

    TFLiteModel(Context context) {
        model = loadModel(context);
        WIDTH = getWidth();
        HEIGHT = getHeight();
    }

    protected ByteBuffer preProcessInput(Bitmap bitmap) {
        int[] pixels = new int[WIDTH * HEIGHT];
        bitmap = Bitmap.createScaledBitmap(bitmap, WIDTH, HEIGHT, true);
        bitmap.getPixels(pixels, 0, WIDTH, 0, 0, WIDTH, HEIGHT);

        ByteBuffer input = ByteBuffer
                .allocateDirect(WIDTH * HEIGHT * 3 * Float.SIZE / Byte.SIZE)
                .order(ByteOrder.nativeOrder());

        for (int px : pixels) {
            float[] rgb = preProcessPixel(px);
            for (float value : rgb)
                input.putFloat(value);
        }

        return input;
    }

    public T run(Bitmap bitmap) {
        long start = SystemClock.uptimeMillis();
        ByteBuffer input = preProcessInput(bitmap);
        long end = SystemClock.uptimeMillis();

        Utils.log("Pre Process Time: " + (end - start));
        Object output = getOutputBuffer();

        start = SystemClock.uptimeMillis();
        model.run(input, output);
        end = SystemClock.uptimeMillis();

        Utils.log("Inference Time: " + (end - start));

        return processOutput(output);
    }

}
