
package com.example.executorchllamademo.nutrilens.database

import androidx.room.Dao
import androidx.room.Delete
import androidx.room.Insert
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface MealDao {

    @Insert
    suspend fun insertMeal(meal: MealEntity): Long

    @Delete
    suspend fun deleteMeal(meal: MealEntity)

    @Query("SELECT * FROM meals ORDER BY timestamp DESC")
    fun getAllMeals(): Flow<List<MealEntity>>

    @Query("""
        SELECT * FROM meals
        WHERE timestamp >= :startTime AND timestamp < :endTime
        ORDER BY timestamp DESC
    """)
    fun getMealsBetween(
        startTime: Long,
        endTime: Long
    ): Flow<List<MealEntity>>
}
