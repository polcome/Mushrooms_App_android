package com.example.mushrooms_app

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView

// Graella de bolets de la pantalla d'inici
class MushroomAdapter(
    private val items: List<Mushroom>,
    private val onClick: (Mushroom) -> Unit,
) : RecyclerView.Adapter<MushroomAdapter.Holder>() {

    class Holder(view: View) : RecyclerView.ViewHolder(view) {
        val image: ImageView = view.findViewById(R.id.image)
        val name: TextView = view.findViewById(R.id.name)
        val commonName: TextView = view.findViewById(R.id.commonName)
        val badge: TextView = view.findViewById(R.id.badge)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): Holder =
        Holder(LayoutInflater.from(parent.context).inflate(R.layout.item_mushroom, parent, false))

    override fun onBindViewHolder(holder: Holder, position: Int) {
        val mushroom = items[position]
        holder.image.setImageResource(mushroom.image)
        holder.name.text = mushroom.label
        holder.commonName.text = mushroom.commonName
        holder.badge.showEdibility(mushroom.edibility)
        holder.itemView.setOnClickListener { onClick(mushroom) }
    }

    override fun getItemCount() = items.size
}
