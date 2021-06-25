package com.example.mushrooms_app

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Color
import android.net.Uri
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import com.example.mushrooms_app.ml.BestModelMobile2
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : AppCompatActivity() {

    lateinit var bitmap:Bitmap
    lateinit var imgview:ImageView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imgview = findViewById(R.id.imageView)

        val fileName = "class_labels.txt"
        val inputString = application.assets.open(fileName).bufferedReader().use { it.readText()}
        var townList = inputString.split("\n")

        var predn:TextView = findViewById(R.id.textView)
        var predp:TextView = findViewById(R.id.textView3)

        var select: Button = findViewById(R.id.button)

        select.setOnClickListener(View.OnClickListener {

            var intent: Intent = Intent(Intent.ACTION_GET_CONTENT)
            intent.type = "image/*"

            startActivityForResult(intent, 100)

        })

        var predict: Button = findViewById(R.id.button2)

        predict.setOnClickListener(View.OnClickListener {


            val model = BestModelMobile2.newInstance(this)

            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)

            val bitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
            val input = ByteBuffer.allocateDirect(224*224*3*4).order(ByteOrder.nativeOrder())
            for (y in 0 until 224) {
                for (x in 0 until 224) {
                    val px = bitmap.getPixel(x, y)

                    // Get channel values from the pixel value.
                    val r = Color.red(px)
                    val g = Color.green(px)
                    val b = Color.blue(px)

                    // Normalize channel values to [-1.0, 1.0]. This requirement depends on the model.
                    // For example, some models might require values to be normalized to the range
                    // [0.0, 1.0] instead.
                    val rf = (r - 127) / 255f
                    val gf = (g - 127) / 255f
                    val bf = (b - 127) / 255f

                    input.putFloat(rf)
                    input.putFloat(gf)
                    input.putFloat(bf)
                }
            }

            val byteBuffer=input

            inputFeature0.loadBuffer(byteBuffer)

            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer

            var max = getMax(outputFeature0.floatArray)
            val predvalue = outputFeature0.floatArray

            predn.setText(townList[max])
            predp.setText(predvalue.get(max).toString())

            model.close()

        })
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        imgview.setImageURI(data?.data)

        var uri: Uri?= data?.data

        bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)

    }

    fun getMax(arr:FloatArray) : Int {

        var ind = 0
        var min = 0.0f

        for (i in 0..11) {
            if (arr[i] > min) {
                ind = i
                min = arr[i]
            }
        }
        return ind
    }


}

