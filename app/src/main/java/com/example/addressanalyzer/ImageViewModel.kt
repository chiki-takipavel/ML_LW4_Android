package com.example.addressanalyzer

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import android.graphics.Bitmap

class ImageViewModel : ViewModel() {
    private val _bitmap = MutableLiveData<Bitmap>()
    val bitmap: LiveData<Bitmap> = _bitmap

    fun setBitmap(bitmap: Bitmap) {
        _bitmap.value = bitmap
    }
}