package com.example.addressanalyzer

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.res.AssetFileDescriptor
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.provider.Settings
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.FileProvider
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import androidx.lifecycle.lifecycleScope
import com.example.addressanalyzer.databinding.FragmentUploaderBinding
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.snackbar.Snackbar
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.TransformToGrayscaleOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.File
import java.io.FileInputStream
import java.io.IOException
import java.nio.channels.FileChannel
import java.text.SimpleDateFormat
import java.util.*
import kotlin.collections.ArrayList

/**
 * A simple [Fragment] subclass as the default destination in the navigation.
 */
class UploaderFragment : Fragment() {

    companion object {
        const val MODEL_FILE_NAME = "model.tflite"
        const val SCALE = 0.5
        const val IMAGE_SIZE = 32
        const val MIN_ASPECT_RATIO = 0.5
        const val MAX_ASPECT_RATIO = 3.0
        const val SOLIDITY_THRESHOLD = 0.4
    }

    private var _binding: FragmentUploaderBinding? = null
    private lateinit var tflite: Interpreter

    // This property is only valid between onCreateView and
    // onDestroyView.
    private val binding get() = _binding!!


    private val viewModel: ImageViewModel by viewModels()
    private val loadFile = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let {
            lifecycleScope.launch(Dispatchers.IO) {
                val source = ImageDecoder.createSource(requireActivity().contentResolver, uri)
                val bitmap = ImageDecoder.decodeBitmap(source)

                val results = findAndRecognizeDigits(bitmap)

                withContext(Dispatchers.Main) {
                    viewModel.setBitmap(results.first)
                    showResultSnackbar("Распознанные числа: ${results.second}")
                }
            }
        }
    }

    private lateinit var currentPhotoPath: String
    private val takePicture = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            lifecycleScope.launch(Dispatchers.IO) {
                val imageFile = File(currentPhotoPath)
                val source = ImageDecoder.createSource(imageFile)
                val bitmap = ImageDecoder.decodeBitmap(source)

                val results = findAndRecognizeDigits(bitmap)

                withContext(Dispatchers.Main) {
                    viewModel.setBitmap(results.first)
                    showResultSnackbar("Распознанные числа: ${results.second}")
                }
            }
        }
    }

    private val requestCameraPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
        if (isGranted) {
            dispatchTakePictureIntent()
        } else {
            showSettingsDialog()
        }
    }

    private fun loadModel(context: Context): Interpreter {
        val assetManager = context.assets
        val fileDescriptor: AssetFileDescriptor = assetManager.openFd(MODEL_FILE_NAME)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel: FileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength

        return Interpreter(fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength))
    }

    private fun findContours(source: Mat): MutableList<MatOfPoint> {
        val newSize = Size(source.width() * SCALE, source.height() * SCALE)
        Imgproc.resize(source, source, newSize)

        val gray = Mat()
        Imgproc.cvtColor(source, gray, Imgproc.COLOR_RGBA2GRAY)

        val blurred = Mat()
        Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0)

        val edges = Mat()
        Imgproc.Canny(blurred, edges, 100.0, 200.0)

        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
        Imgproc.morphologyEx(edges, edges, Imgproc.MORPH_CLOSE, kernel)

        val contours: MutableList<MatOfPoint> = ArrayList()
        val hierarchy = Mat()
        Imgproc.findContours(
            edges,
            contours,
            hierarchy,
            Imgproc.RETR_EXTERNAL,
            Imgproc.CHAIN_APPROX_SIMPLE
        )

        return contours
    }

    private fun transformRectToSquare(source: Mat, rect: Rect): Rect {
        val maxSide = kotlin.math.max(rect.width, rect.height)
        val dx = maxSide - rect.width
        val dy = maxSide - rect.height
        val newX = kotlin.math.max(0, rect.x - dx / 2)
        val newY = kotlin.math.max(0, rect.y - dy / 2)
        val newWidth = if (newX + maxSide > source.width()) source.width() - newX else maxSide
        val newHeight = if (newY + maxSide > source.height()) source.height() - newY else maxSide

        return Rect(newX, newY, newWidth, newHeight)
    }

    private suspend fun findAndRecognizeDigits(bitmap: Bitmap): Pair<Bitmap, List<Int>> {
        return withContext(Dispatchers.Default) {
            val convertedBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
            val source = Mat()
            Utils.bitmapToMat(convertedBitmap, source)

            val recognizedDigits = ArrayList<Int>()
            val contours = findContours(source)
            for (contour in contours) {
                val rect = Imgproc.boundingRect(contour)
                if (rect.width < 6 || rect.height < 6) continue

                val aspectRatio = rect.width.toDouble() / rect.height.toDouble()
                if (aspectRatio !in MIN_ASPECT_RATIO..MAX_ASPECT_RATIO) continue

                val area = Imgproc.contourArea(contour)
                val boundingRectArea = rect.width * rect.height
                val solidity = area / boundingRectArea
                if (solidity < SOLIDITY_THRESHOLD) continue

                val newRect = transformRectToSquare(source, rect)
                val digitMat = Mat(source, newRect)
                val digitBitmap = Bitmap.createBitmap(digitMat.cols(), digitMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(digitMat, digitBitmap)

                val recognizedDigit = recognizeImage(digitBitmap)
                recognizedDigits.add(recognizedDigit)

                Imgproc.rectangle(source, newRect.tl(), newRect.br(), Scalar(255.0, 0.0, 0.0), 2)
            }

            val resultBitmap = Bitmap.createBitmap(source.cols(), source.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(source, resultBitmap)

            Pair(resultBitmap, recognizedDigits)
        }
    }

    private fun recognizeImage(bitmap: Bitmap): Int {
        val convertedBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)

        val inputProcessor = ImageProcessor.Builder()
            .add(ResizeOp(IMAGE_SIZE, IMAGE_SIZE, ResizeOp.ResizeMethod.BILINEAR))
            .add(TransformToGrayscaleOp())
            .add(NormalizeOp(0f, 255f))
            .build()
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(convertedBitmap)

        val inputBuffer = inputProcessor.process(tensorImage)
        val outputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 10), DataType.FLOAT32)

        tflite.run(inputBuffer.buffer, outputBuffer.buffer)

        return selectClass(outputBuffer.floatArray)
    }

    private fun selectClass(results: FloatArray, threshold: Float = 0.5f): Int {
        var maxIndex = 0
        var maxValue = results[0]
        for (i in 1 until results.size) {
            if (results[i] > maxValue) {
                maxValue = results[i]
                maxIndex = i
            }
        }

        return if (maxValue >= threshold) maxIndex else -1
    }

    override fun onCreateView(
            inflater: LayoutInflater, container: ViewGroup?,
            savedInstanceState: Bundle?
    ): View {

        _binding = FragmentUploaderBinding.inflate(inflater, container, false)
        return binding.root

    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        viewModel.bitmap.observe(viewLifecycleOwner) { bitmap ->
            binding.imageviewPhoto.setImageBitmap(bitmap)
        }

        binding.buttonChoose.setOnClickListener {
            loadFile.launch("image/*")
        }

        binding.fab.setOnClickListener {
            requestCameraPermission.launch(Manifest.permission.CAMERA)
        }

        tflite = loadModel(requireContext())
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
        tflite.close()
    }

    @Throws(IOException::class)
    private fun createImageFile(): File {
        val timeStamp: String = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        val storageDir: File? = requireContext().getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        return File.createTempFile("JPEG_${timeStamp}_", ".jpg", storageDir).apply {
            currentPhotoPath = absolutePath
        }
    }

    private fun dispatchTakePictureIntent() {
        Intent(MediaStore.ACTION_IMAGE_CAPTURE).also { takePictureIntent ->
            takePictureIntent.resolveActivity(requireActivity().packageManager)?.also {
                val photoFile: File? = try {
                    createImageFile()
                } catch (ex: IOException) {
                    null
                }
                photoFile?.also {
                    val photoURI: Uri = FileProvider.getUriForFile(
                        requireContext(),
                        "${requireContext().packageName}.fileprovider",
                        it
                    )
                    takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI)
                    takePicture.launch(takePictureIntent)
                }
            }
        }
    }

    private fun showSettingsDialog() {
        MaterialAlertDialogBuilder(requireContext(), com.google.android.material.R.style.ThemeOverlay_Material3_MaterialAlertDialog_Centered)
            .setIcon(R.drawable.ic_videocam)
            .setTitle(resources.getString(R.string.camera_dialog_title))
            .setMessage(resources.getString(R.string.camera_dialog_description))
            .setPositiveButton(resources.getString(R.string.camera_dialog_positive)) { _, _ ->
                val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS)
                val uri = Uri.fromParts("package", requireActivity().packageName, null)
                intent.data = uri
                startActivity(intent)
            }
            .setNegativeButton(resources.getString(R.string.camera_dialog_negative)) { dialog, _ ->
                dialog.dismiss()
            }
            .show()
    }

    private fun showResultSnackbar(text: String) {
        val snackbarResult = Snackbar.make(binding.root, text, Snackbar.LENGTH_INDEFINITE)
        snackbarResult.anchorView = binding.fab
        snackbarResult.setAction(R.string.button_ok) {
            snackbarResult.dismiss()
        }
        snackbarResult.show()
    }
}