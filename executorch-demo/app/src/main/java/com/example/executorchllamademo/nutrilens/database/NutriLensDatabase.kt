
package com.example.executorchllamademo.nutrilens.database

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase

@Database(
    entities = [MealEntity::class],
    version = 1,
    exportSchema = false
)
abstract class NutriLensDatabase : RoomDatabase() {

    abstract fun mealDao(): MealDao

    companion object {
        @Volatile
        private var INSTANCE: NutriLensDatabase? = null

        fun getDatabase(context: Context): NutriLensDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    NutriLensDatabase::class.java,
                    "nutrilens_database"
                ).build()

                INSTANCE = instance
                instance
            }
        }
    }
}
