package com.example.mushrooms_app

import android.animation.ValueAnimator
import android.content.ActivityNotFoundException
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.view.HapticFeedbackConstants
import android.view.LayoutInflater
import android.view.View
import android.view.animation.DecelerateInterpolator
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.core.widget.NestedScrollView
import androidx.exifinterface.media.ExifInterface
import com.example.mushrooms_app.ml.BestModelMobile2
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.bottomsheet.BottomSheetBehavior
import com.google.android.material.bottomsheet.BottomSheetDialog
import com.google.android.material.button.MaterialButton
import com.google.android.material.card.MaterialCardView
import com.google.android.material.progressindicator.LinearProgressIndicator
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors
import kotlin.math.roundToInt

class MainActivity : AppCompatActivity() {

    private lateinit var scrollView: NestedScrollView
    private lateinit var imageView: ImageView
    private lateinit var placeholder: View
    private lateinit var clearButton: MaterialButton
    private lateinit var galleryButton: MaterialButton
    private lateinit var cameraButton: MaterialButton
    private lateinit var predictButton: MaterialButton
    private lateinit var progress: LinearProgressIndicator
    private lateinit var resultCard: MaterialCardView
    private lateinit var resultName: TextView
    private lateinit var confidenceBadge: TextView
    private lateinit var resultBar: LinearProgressIndicator
    private lateinit var resultPercent: TextView
    private lateinit var otherTitle: TextView
    private lateinit var otherResults: LinearLayout
    private lateinit var moreInfoButton: MaterialButton

    private lateinit var labels: List<String>
    private var bitmap: Bitmap? = null
    private var resultLabel: String? = null
    private var busy = false

    // El model només es fa servir des d'aquest fil, així la pantalla no es bloqueja
    private val executor = Executors.newSingleThreadExecutor()
    private var model: BestModelMobile2? = null

    private val pickImage = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) loadImage(uri)
    }

    private val takePicture = registerForActivityResult(ActivityResultContracts.TakePicture()) { success ->
        if (success) loadImage(cameraUri())
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        scrollView = findViewById(R.id.scrollView)
        imageView = findViewById(R.id.imageView)
        placeholder = findViewById(R.id.placeholder)
        clearButton = findViewById(R.id.clearButton)
        galleryButton = findViewById(R.id.galleryButton)
        cameraButton = findViewById(R.id.cameraButton)
        predictButton = findViewById(R.id.predictButton)
        progress = findViewById(R.id.progress)
        resultCard = findViewById(R.id.resultCard)
        resultName = findViewById(R.id.resultName)
        confidenceBadge = findViewById(R.id.confidenceBadge)
        resultBar = findViewById(R.id.resultBar)
        resultPercent = findViewById(R.id.resultPercent)
        otherTitle = findViewById(R.id.otherTitle)
        otherResults = findViewById(R.id.otherResults)
        moreInfoButton = findViewById(R.id.moreInfoButton)

        labels = assets.open("class_labels.txt").bufferedReader().use { it.readLines() }
            .map { it.trim() }
            .filter { it.isNotEmpty() }

        findViewById<View>(R.id.imageCard).setOnClickListener { if (!busy) openGallery() }
        galleryButton.setOnClickListener { openGallery() }
        cameraButton.setOnClickListener { openCamera() }
        clearButton.setOnClickListener { clearImage() }
        predictButton.setOnClickListener { predict() }
        moreInfoButton.setOnClickListener { resultLabel?.let { MushroomCatalog.find(it) }?.let { showMushroom(it) } }

        findViewById<RecyclerView>(R.id.mushroomList).adapter = MushroomAdapter(MushroomCatalog.all) { showMushroom(it) }

        if (!packageManager.hasSystemFeature(PackageManager.FEATURE_CAMERA_ANY)) {
            cameraButton.visibility = View.GONE
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        executor.execute { model?.close() }
        executor.shutdown()
    }

    // ---------- Triar la foto ----------

    private fun openGallery() {
        pickImage.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
    }

    private fun openCamera() {
        try {
            takePicture.launch(cameraUri())
        } catch (e: ActivityNotFoundException) {
            toast(R.string.error_no_camera)
        }
    }

    // On la càmera desa la foto (memòria cau de l'app, compartida via FileProvider)
    private fun cameraUri(): Uri {
        val dir = File(cacheDir, "images").apply { mkdirs() }
        return FileProvider.getUriForFile(this, "$packageName.fileprovider", File(dir, "camera.jpg"))
    }

    private fun loadImage(uri: Uri) {
        setBusy(true)
        executor.execute {
            val loaded = try {
                decodeImage(uri)
            } catch (e: Exception) {
                null
            }
            runOnUiThread {
                if (isDestroyed) return@runOnUiThread
                setBusy(false)
                if (loaded == null) {
                    toast(R.string.error_image)
                } else {
                    showImage(loaded)
                }
            }
        }
    }

    // Llegeix la imatge reduïda (per no gastar massa memòria) i girada segons l'EXIF
    private fun decodeImage(uri: Uri): Bitmap? {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }

        var sample = 1
        while (bounds.outWidth / (sample * 2) >= MAX_IMAGE_SIDE && bounds.outHeight / (sample * 2) >= MAX_IMAGE_SIDE) {
            sample *= 2
        }
        val options = BitmapFactory.Options().apply { inSampleSize = sample }
        val decoded = contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, options) }
            ?: return null

        val rotation = try {
            contentResolver.openInputStream(uri)?.use { ExifInterface(it).rotationDegrees } ?: 0
        } catch (e: Exception) {
            0
        }
        if (rotation == 0) return decoded

        val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
        return Bitmap.createBitmap(decoded, 0, 0, decoded.width, decoded.height, matrix, true)
    }

    private fun showImage(loaded: Bitmap) {
        bitmap = loaded
        hideResult()

        imageView.alpha = 0f
        imageView.scaleX = 1.05f
        imageView.scaleY = 1.05f
        imageView.setImageBitmap(loaded)
        imageView.animate().alpha(1f).scaleX(1f).scaleY(1f).setDuration(350)
            .setInterpolator(DecelerateInterpolator()).start()

        placeholder.animate().alpha(0f).setDuration(200)
            .withEndAction { placeholder.visibility = View.GONE }.start()

        clearButton.visibility = View.VISIBLE
        predictButton.isEnabled = true
    }

    private fun clearImage() {
        bitmap = null
        hideResult()
        imageView.setImageDrawable(null)
        placeholder.visibility = View.VISIBLE
        placeholder.alpha = 0f
        placeholder.animate().alpha(1f).setDuration(250).start()
        clearButton.visibility = View.GONE
        predictButton.isEnabled = false
    }

    // ---------- Identificar ----------

    private fun predict() {
        val selected = bitmap ?: return
        setBusy(true)
        hideResult()

        executor.execute {
            val scores = try {
                classify(selected)
            } catch (e: Exception) {
                null
            }
            runOnUiThread {
                if (isDestroyed) return@runOnUiThread
                setBusy(false)
                if (scores == null) {
                    toast(R.string.error_model)
                } else {
                    showResult(scores)
                }
            }
        }
    }

    private fun classify(selected: Bitmap): FloatArray {
        val m = model ?: BestModelMobile2.newInstance(this).also { model = it }

        val scaled = Bitmap.createScaledBitmap(selected, INPUT_SIZE, INPUT_SIZE, true)
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        scaled.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        val input = ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * 3 * 4).order(ByteOrder.nativeOrder())
        for (px in pixels) {
            // Normalitza a [-1.0, 1.0], igual que mobilenet.preprocess_input
            // amb què es va entrenar el model (TFG_MobileNet.ipynb)
            input.putFloat(Color.red(px) / 127.5f - 1f)
            input.putFloat(Color.green(px) / 127.5f - 1f)
            input.putFloat(Color.blue(px) / 127.5f - 1f)
        }

        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, INPUT_SIZE, INPUT_SIZE, 3), DataType.FLOAT32)
        inputFeature0.loadBuffer(input)

        return m.process(inputFeature0).outputFeature0AsTensorBuffer.floatArray
    }

    // ---------- Mostrar el resultat ----------

    private fun showResult(scores: FloatArray) {
        val ranked = scores.indices.sortedByDescending { scores[it] }
        val best = ranked.first()
        val bestPercent = scores[best] * 100

        resultLabel = labelFor(best)
        resultName.text = resultLabel
        moreInfoButton.visibility = if (MushroomCatalog.find(labelFor(best)) != null) View.VISIBLE else View.GONE

        val (levelText, levelColor) = when {
            bestPercent >= 70 -> R.string.confidence_high to R.color.conf_high
            bestPercent >= 40 -> R.string.confidence_mid to R.color.conf_mid
            else -> R.string.confidence_low to R.color.conf_low
        }
        confidenceBadge.setText(levelText)
        confidenceBadge.backgroundTintList = ColorStateList.valueOf(ContextCompat.getColor(this, levelColor))

        resultBar.setProgressCompat(0, false)
        resultPercent.text = getString(R.string.result_percent, 0f)

        otherResults.removeAllViews()
        val others = ranked.drop(1).take(OTHER_RESULTS)
        val otherBars = others.map { index ->
            val item = LayoutInflater.from(this).inflate(R.layout.item_prediction, otherResults, false)
            item.findViewById<TextView>(R.id.name).text = labelFor(index)
            item.findViewById<TextView>(R.id.percent).text = getString(R.string.percent_short, scores[index] * 100)
            otherResults.addView(item)
            item.findViewById<LinearProgressIndicator>(R.id.bar) to scores[index] * 100
        }
        otherTitle.visibility = if (others.isEmpty()) View.GONE else View.VISIBLE

        // Entrada de la targeta i barres que s'omplen
        resultCard.visibility = View.VISIBLE
        resultCard.alpha = 0f
        resultCard.translationY = resources.displayMetrics.density * 40
        resultCard.animate().alpha(1f).translationY(0f).setDuration(400)
            .setInterpolator(DecelerateInterpolator())
            .withEndAction {
                resultBar.setProgressCompat(bestPercent.roundToInt(), true)
                otherBars.forEach { (bar, percent) -> bar.setProgressCompat(percent.roundToInt(), true) }
            }
            .start()

        ValueAnimator.ofFloat(0f, bestPercent).apply {
            duration = 900
            startDelay = 300
            interpolator = DecelerateInterpolator()
            addUpdateListener { resultPercent.text = getString(R.string.result_percent, it.animatedValue as Float) }
            start()
        }

        resultCard.performHapticFeedback(HapticFeedbackConstants.KEYBOARD_TAP)
        scrollView.post { scrollView.smoothScrollTo(0, resultCard.top) }
    }

    private fun hideResult() {
        resultCard.animate().cancel()
        resultCard.visibility = View.GONE
    }

    // ---------- Fitxa d'un bolet ----------

    private fun showMushroom(mushroom: Mushroom) {
        val view = layoutInflater.inflate(R.layout.sheet_mushroom, null)
        view.findViewById<ImageView>(R.id.image).setImageResource(mushroom.image)
        view.findViewById<TextView>(R.id.name).text = mushroom.label
        view.findViewById<TextView>(R.id.commonName).text = mushroom.commonName
        view.findViewById<TextView>(R.id.badge).showEdibility(mushroom.edibility)
        view.findViewById<TextView>(R.id.description).text = mushroom.description
        view.findViewById<TextView>(R.id.credit).text = getString(R.string.photo_credit, mushroom.photoCredit)

        val features = view.findViewById<LinearLayout>(R.id.features)
        mushroom.features.forEach { feature ->
            val row = layoutInflater.inflate(R.layout.item_feature, features, false)
            row.findViewById<TextView>(R.id.text).text = feature
            features.addView(row)
        }

        BottomSheetDialog(this).apply {
            setContentView(view)
            behavior.skipCollapsed = true
            behavior.state = BottomSheetBehavior.STATE_EXPANDED
            show()
        }
    }

    // ---------- Utilitats ----------

    private fun setBusy(value: Boolean) {
        busy = value
        progress.visibility = if (value) View.VISIBLE else View.GONE
        galleryButton.isEnabled = !value
        cameraButton.isEnabled = !value
        clearButton.isEnabled = !value
        predictButton.isEnabled = !value && bitmap != null
        predictButton.setText(if (value) R.string.analyzing else R.string.identify)
    }

    private fun labelFor(index: Int) = labels.getOrElse(index) { "?" }

    private fun toast(message: Int) = Toast.makeText(this, message, Toast.LENGTH_SHORT).show()

    companion object {
        private const val INPUT_SIZE = 224
        private const val MAX_IMAGE_SIDE = 1024
        private const val OTHER_RESULTS = 3
    }

}
