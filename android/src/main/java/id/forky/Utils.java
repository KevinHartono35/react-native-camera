package id.forky;

import android.content.Context;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class Utils {

    public static void log(String message) {
        Log.e("My Log", message);
    }

    public static Interpreter loadModelFromAssets(Context ctx, String fileName) {
        try {
            InputStream inputStream = ctx.getAssets().open(fileName);
            byte[] model = new byte[inputStream.available()];
            int result = inputStream.read(model);
            if (result == -1)
                return null;
            ByteBuffer buffer = ByteBuffer.allocateDirect(model.length)
                    .order(ByteOrder.nativeOrder());
            buffer.put(model);
            return new Interpreter(buffer);
        } catch (IOException e) {
            return null;
        }
    }

}
