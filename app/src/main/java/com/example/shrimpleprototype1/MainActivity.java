/*
 * Created by ishaanjav
 * github.com/ishaanjav
*/

package com.example.shrimpleprototype1;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.schema.Model;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import java.io.IOException;
import com.example.shrimpleprototype1.ml.Modelquant;
import org.tensorflow.lite.Interpreter;


import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    TextView result, confidence;
    ImageView imageView;
    Button picture;
    int imageSize = 640;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        result = findViewById(R.id.result);
        confidence = findViewById(R.id.confidence);
        imageView = findViewById(R.id.imageView);
        picture = findViewById(R.id.button);

        picture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Launch camera if we have permission
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 1);
                } else {
                    //Request camera permission if we don't have it.
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });
    }

    public void classifyImage(Bitmap image) {
        try {
            Modelquant model = Modelquant.newInstance(context);

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 640, 640, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());
            inputFeature0.loadBuffer(byteBuffer);
            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            int pixel = 0;
            for (int i = 0; i < imageSize; i++) {
                for (int j = 0; j < imageSize; j++) {
                    int val = intValues[pixel++];
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));
                    // Runs model inference and gets result.
                    inputFeature0.loadBuffer(byteBuffer);

                    // Runs model inference and gets result.
                    Modelquant.Outputs outputs = model.process(inputFeature0);
                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                    float[] confidences = outputFeature0.getFloatArray();
                    int maxPos = 0;
                    float maxConfidence = 0;
                    for (int x = 0; x < confidences.length; x++) {
                        if (confidences[x] > maxConfidence) {
                            maxConfidence = confidences[x];
                            maxPos = x;
                        }
                    }
                    String[] classes = {"Black Gill", "Normal", "Vibriosis", "WSSV"};

                    result.setText(classes[maxPos]);

                    String s = "";
                    for (int y = 0; y < classes.length; y++) {
                        s += String.format("%s: %.1f%%\n", classes[y], confidences[i] * 100);
                    }
                    confidence.setText(s);
                    // better naming convention of x, y, i and j variables
                    // Releases model resources if no longer used.
                    model.close();
                }
            }
        } catch (Exception e) {
            // TODO Handle the exception
            // Better handling
        }
    }
    @Override
    public void onActivityResult ( int requestCode, int resultCode, @Nullable Intent data){
        if (requestCode == 1 && resultCode == RESULT_OK) {
            Bitmap image = (Bitmap) data.getExtras().get("data");
            int dimension = Math.min(image.getWidth(), image.getHeight());
            image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
            imageView.setImageBitmap(image);

            image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
            classifyImage(image);
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}




