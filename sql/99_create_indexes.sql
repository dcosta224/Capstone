-- Indexes for common joins (run after data load)
SET search_path TO usda, public;

CREATE INDEX IF NOT EXISTS idx_food_data_type ON food (data_type);
CREATE INDEX IF NOT EXISTS idx_food_category_id ON food (food_category_id);

CREATE INDEX IF NOT EXISTS idx_branded_food_gtin ON branded_food (gtin_upc);
CREATE INDEX IF NOT EXISTS idx_branded_food_brand_owner ON branded_food (brand_owner);

CREATE INDEX IF NOT EXISTS idx_food_nutrient_fdc_id ON food_nutrient (fdc_id);
CREATE INDEX IF NOT EXISTS idx_food_nutrient_nutrient_id ON food_nutrient (nutrient_id);
CREATE INDEX IF NOT EXISTS idx_food_nutrient_fdc_nutrient ON food_nutrient (fdc_id, nutrient_id);

CREATE INDEX IF NOT EXISTS idx_food_portion_fdc_id ON food_portion (fdc_id);
CREATE INDEX IF NOT EXISTS idx_food_component_fdc_id ON food_component (fdc_id);
CREATE INDEX IF NOT EXISTS idx_input_food_fdc_id ON input_food (fdc_id);

CREATE INDEX IF NOT EXISTS idx_fndds_ingredient_code ON fndds_ingredient_nutrient_value (ingredient_code);
CREATE INDEX IF NOT EXISTS idx_fndds_fdc_id ON fndds_ingredient_nutrient_value (fdc_id);

CREATE INDEX IF NOT EXISTS idx_sub_sample_food_sample ON sub_sample_food (fdc_id_of_sample_food);
CREATE INDEX IF NOT EXISTS idx_sub_sample_result_food_nutrient ON sub_sample_result (food_nutrient_id);

ANALYZE;
