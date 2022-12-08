package com.example.camerax.face_detection

import android.content.ContentValues.TAG
import android.graphics.Rect
import android.util.Log
import com.example.camerax.BaseIMageAnalyzer
import com.example.camerax.FaceContourGraphic
import com.example.camerax.GraphicOverlay
import com.google.android.gms.tasks.Task
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import java.io.IOException

class FaceContourDetectionProcessor(private val view: GraphicOverlay) :
    BaseIMageAnalyzer<List<Face>>() {

    private val realTimeOptions = FaceDetectorOptions.Builder()
        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
        .setContourMode(FaceDetectorOptions.CONTOUR_MODE_NONE)
        .build()

    private val detector = FaceDetection.getClient(realTimeOptions)

    override val graphicOverlay: GraphicOverlay
        get() = view

    override fun detectInImage(image: InputImage): Task<List<Face>> {
        return detector.process(image)
    }

    override fun stop() {
        try {
            detector.close()
        } catch (e: IOException) {
            Log.e(TAG, "Exception thrown while closing Face Detector")
        }
    }

    override fun onSuccess(results: List<Face>, graphicOverlay: GraphicOverlay, rect: Rect) {
        graphicOverlay.clear()
        results.forEach {
            val faceGraphic = FaceContourGraphic(graphicOverlay, it, rect)
            graphicOverlay.add(faceGraphic)
        }
        graphicOverlay.postInvalidate()
    }

    override fun onFailure(e: Exception) {
        Log.w(TAG, "Face Detector failed")
    }
}