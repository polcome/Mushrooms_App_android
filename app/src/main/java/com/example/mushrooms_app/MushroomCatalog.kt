package com.example.mushrooms_app

import android.content.res.ColorStateList
import android.widget.TextView
import androidx.annotation.ColorRes
import androidx.annotation.DrawableRes
import androidx.annotation.StringRes
import androidx.core.content.ContextCompat

enum class Edibility(@StringRes val text: Int, @ColorRes val color: Int) {
    MOSTLY_EDIBLE(R.string.edibility_mostly_edible, R.color.conf_high),
    MIXED(R.string.edibility_mixed, R.color.conf_mid),
    TOXIC(R.string.edibility_toxic, R.color.conf_low),
    DEADLY(R.string.edibility_deadly, R.color.conf_low),
    NO_INTEREST(R.string.edibility_no_interest, R.color.md_outline),
}

data class Mushroom(
    val label: String,
    val commonName: String,
    @DrawableRes val image: Int,
    val description: String,
    val features: List<String>,
    val edibility: Edibility,
    val photoCredit: String,
)

// Fitxa de cada gènere que reconeix el model (mateix nom que a class_labels.txt).
// Fotos de Wikimedia Commons, amb llicència lliure.
object MushroomCatalog {

    val all = listOf(
        Mushroom(
            label = "Agaricus",
            commonName = "Xampinyons, camperols",
            image = R.drawable.mush_agaricus,
            description = "Bolets de barret blanc o marronós, amb anell al peu i làmines que comencen rosades i acaben de color marró xocolata. Inclou el xampinyó de cultiu i el camperol (A. campestris). Alguns, com A. xanthodermus, són tòxics: es tornen grocs en fregar-los i fan olor de tinta.",
            features = listOf("Làmines rosades que passen a marró fosc", "Anell al peu", "Sense volva a la base"),
            edibility = Edibility.MIXED,
            photoCredit = "Alan Rockefeller · CC BY-SA 4.0",
        ),
        Mushroom(
            label = "Amanita",
            commonName = "Reig bord, farinera borda, ou de reig",
            image = R.drawable.mush_amanita,
            description = "El gènere més perillós: inclou la farinera borda (A. phalloides), responsable de la majoria de morts per bolets. També hi ha el reig bord (A. muscaria), vermell amb taques blanques, i l'ou de reig (A. caesarea), un comestible molt apreciat que es pot confondre amb espècies tòxiques.",
            features = listOf("Volva (una mena de sac) a la base del peu", "Anell sota el barret", "Làmines blanques"),
            edibility = Edibility.DEADLY,
            photoCredit = "Onderwijsgek · CC BY-SA 3.0 NL",
        ),
        Mushroom(
            label = "Boletus",
            commonName = "Ceps",
            image = R.drawable.mush_boletus,
            description = "Bolets carnosos amb porus sota el barret, com una esponja, en lloc de làmines, i un peu gruixut, sovint amb una xarxa dibuixada. Inclou el cep (B. edulis), un dels bolets més apreciats. Alguns bolets semblants amb porus vermellosos, com el mataparent (Rubroboletus satanas), són tòxics.",
            features = listOf("Porus en lloc de làmines", "Peu gruixut i panxut", "Carn ferma i blanca"),
            edibility = Edibility.MOSTLY_EDIBLE,
            photoCredit = "Tocekas · CC BY-SA 3.0",
        ),
        Mushroom(
            label = "Cortinarius",
            commonName = "Cortinaris",
            image = R.drawable.mush_cortinarius,
            description = "El gènere de bolets amb làmines més gran, amb milers d'espècies molt difícils de distingir. De joves tenen una cortina, un vel com una teranyina entre el barret i el peu. Alguns, com C. orellanus, són mortals i danyen els ronyons dies després de menjar-los.",
            features = listOf("Cortina (vel de teranyina) quan és jove", "Espores de color rovell", "Colors molt variats: lila, taronja, marró"),
            edibility = Edibility.DEADLY,
            photoCredit = "JJ Harrison · CC BY-SA 3.0",
        ),
        Mushroom(
            label = "Entoloma",
            commonName = "Entolomes",
            image = R.drawable.mush_entoloma,
            description = "Bolets amb làmines que es tornen rosades a mesura que maduren. L'espècie més coneguda, E. sinuatum (la de la foto), és molt tòxica i causa intoxicacions digestives fortes. Es pot confondre amb espècies comestibles de prat.",
            features = listOf("Làmines rosades en madurar", "Sense anell ni volva", "Algunes fan olor de farina"),
            edibility = Edibility.TOXIC,
            photoCredit = "Archenzo (retall: Ak ccm) · CC BY-SA 3.0",
        ),
        Mushroom(
            label = "Exidia",
            commonName = "Gelatines negres",
            image = R.drawable.mush_exidia,
            description = "Fongs gelatinosos que creixen sobre branques i troncs morts, sovint de roure. E. glandulosa forma masses negres i arrugades que s'assequen quan no plou i es tornen a inflar amb la humitat. No són tòxics, però no tenen interès a la cuina.",
            features = listOf("Textura gelatinosa", "Sobre fusta morta", "Sense barret ni peu"),
            edibility = Edibility.NO_INTEREST,
            photoCredit = "Dan Molter (Mushroom Observer) · CC BY-SA 3.0",
        ),
        Mushroom(
            label = "Hygrocybe",
            commonName = "Higròcibes",
            image = R.drawable.mush_hygrocybe,
            description = "Bolets petits de colors molt vius (vermell, taronja, groc o fins i tot verd) i aspecte de cera. Creixen sobretot en prats vells sense adobs i indiquen que el prat està ben conservat. Alguns són lleugerament tòxics.",
            features = listOf("Colors vius i brillants", "Làmines gruixudes i separades", "Barret humit o viscós"),
            edibility = Edibility.NO_INTEREST,
            photoCredit = "Usuari de Mushroom Observer (obs. 69428) · CC BY-SA 3.0",
        ),
        Mushroom(
            label = "Inocybe",
            commonName = "Inocibes",
            image = R.drawable.mush_inocybe,
            description = "Bolets petits i marronosos, amb el barret fibrós, sovint cònic i esquerdat des del centre cap a fora. Molts contenen muscarina, que provoca suors, salivació i problemes digestius poc després de menjar-los. Cap no s'ha de menjar.",
            features = listOf("Barret cònic i fibrós", "Colors marró o crema", "Olor de terra"),
            edibility = Edibility.TOXIC,
            photoCredit = "Eric Steinert · CC BY-SA 3.0",
        ),
        Mushroom(
            label = "Lactarius",
            commonName = "Rovellons, pinetells",
            image = R.drawable.mush_lactarius,
            description = "Quan es trenquen deixen anar un làtex, una mena de llet, de color variable. Inclou bolets molt buscats com el rovelló (L. deliciosus), de làtex taronja. D'altres, com L. torminosus (el de la foto), tenen làtex blanc, són picants i provoquen trastorns digestius.",
            features = listOf("Treu làtex quan es trenca", "Barret sovint amb cercles concèntrics", "Carn que es trenca com un guix"),
            edibility = Edibility.MIXED,
            photoCredit = "Th. Kuhnigk · CC BY 3.0",
        ),
        Mushroom(
            label = "Pluteus",
            commonName = "Bolets de fusta",
            image = R.drawable.mush_pluteus,
            description = "Bolets que creixen sobre fusta morta o restes de fusta. Tenen les làmines lliures, sense tocar el peu, que passen de blanques a rosades. La majoria són comestibles de poca qualitat i alguns, com P. salicinus, són al·lucinògens.",
            features = listOf("Sobre fusta morta", "Làmines lliures, de blanques a rosades", "Sense anell ni volva"),
            edibility = Edibility.NO_INTEREST,
            photoCredit = "James Lindsey · CC BY-SA 3.0",
        ),
        Mushroom(
            label = "Russula",
            commonName = "Llores, rúsules",
            image = R.drawable.mush_russula,
            description = "Bolets de colors molt variats (vermell, verd, lila, groc) amb una carn que es trenca com un guix i no treu làtex. Inclou bons comestibles com la llora (R. cyanoxantha), però d'altres són picants i irritants, com R. emetica, que provoca vòmits.",
            features = listOf("Carn que es trenca com un guix", "Sense làtex", "Barret de colors vius"),
            edibility = Edibility.MIXED,
            photoCredit = "Alan Rockefeller · CC BY-SA 4.0",
        ),
        Mushroom(
            label = "Suillus",
            commonName = "Bolets de pi",
            image = R.drawable.mush_suillus,
            description = "Bolets amb porus sota el barret que viuen associats a coníferes, sobretot pins. El barret és enganxós quan és humit. Molts són comestibles, com S. luteus, tot i que convé treure'n la pell del barret i a algunes persones els poden causar molèsties digestives.",
            features = listOf("Porus grocs sota el barret", "Barret viscós", "Sempre a prop de pins"),
            edibility = Edibility.MOSTLY_EDIBLE,
            photoCredit = "Jerzy Opioła · CC BY-SA 3.0",
        ),
    )

    fun find(label: String): Mushroom? = all.firstOrNull { it.label.equals(label, ignoreCase = true) }
}

fun TextView.showEdibility(edibility: Edibility) {
    setText(edibility.text)
    backgroundTintList = ColorStateList.valueOf(ContextCompat.getColor(context, edibility.color))
}
