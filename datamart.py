# Databricks notebook source
# MAGIC %sql
# MAGIC -- TODO: move to joyce persona volume
# MAGIC CREATE
# MAGIC OR REPLACE TEMPORARY VIEW jb_subcat USING CSV OPTIONS (
# MAGIC     path "/mnt/prd/crm_dashboard/imx/JB_subcat.csv",
# MAGIC     header "true",
# MAGIC     mode "PERMISSIVE"
# MAGIC );

# COMMAND ----------

# MAGIC %md
# MAGIC ## transaction data

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE
# MAGIC OR REPLACE TEMPORARY VIEW vip AS
# MAGIC SELECT
# MAGIC   DISTINCT vip_no,
# MAGIC   vip_main_no
# MAGIC FROM
# MAGIC   imx_prd.imx_dw_train_silver.dbo_viw_lc_sales_vip_masked
# MAGIC WHERE
# MAGIC   isnull(vip_main_no) = 0;
# MAGIC
# MAGIC CREATE
# MAGIC OR REPLACE TEMPORARY VIEW RawSales AS
# MAGIC SELECT
# MAGIC   DISTINCT
# MAGIC   region_key,
# MAGIC   vip_no,
# MAGIC   COALESCE(v.vip_main_no, vip_no) AS vip_main_no, -- if the vip_no is not in the vip table, use the vip_no in the sale table as vip_main_no
# MAGIC   shop_code,
# MAGIC   invoice_no,
# MAGIC   item_code,
# MAGIC   CAST(sales_date AS DATE) AS order_date,
# MAGIC   CAST(sold_qty AS INT) AS sold_qty,
# MAGIC   CAST(net_amt_hkd AS DECIMAL) AS net_amt_hkd,
# MAGIC   CAST(net_amt AS DECIMAL) AS net_amt,
# MAGIC   CAST(item_list_price AS DECIMAL) AS item_list_price,
# MAGIC   shop_brand,
# MAGIC   prod_brand,
# MAGIC   sale_lady_id,
# MAGIC   cashier_id,
# MAGIC   customer_nat AS customer_nat,
# MAGIC   cust_nat_cat AS cust_nat_cat,
# MAGIC   customer_sex AS customer_sex,
# MAGIC   customer_age_group AS customer_age_group,
# MAGIC   void_flag,
# MAGIC   valid_tx_flag,
# MAGIC   sales_staff_flag,
# MAGIC   concat(shop_code, '-', invoice_no) AS sales_main_key, -- use this to identify order
# MAGIC   sales_type
# MAGIC FROM
# MAGIC   imx_prd.imx_dw_train_silver.dbo_viw_lc_sales_sales
# MAGIC   LEFT JOIN vip v USING (vip_no);

# COMMAND ----------

# trim all columns, imx sales sales is not trimmed
from pyspark.sql.functions import trim

df = spark.table("RawSales")
for column in df.columns:
    df = df.withColumn(column, trim(df[column]))

df.createOrReplaceTempView("RawSales")

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE
# MAGIC OR REPLACE TEMPORARY VIEW JoyceBeautySales AS 
# MAGIC WITH qty_table AS (
# MAGIC   SELECT
# MAGIC     sales_main_key,
# MAGIC     SUM(sold_qty) AS total_qty
# MAGIC   FROM
# MAGIC     RawSales
# MAGIC   GROUP BY
# MAGIC     sales_main_key
# MAGIC )
# MAGIC SELECT
# MAGIC   a.*
# MAGIC FROM
# MAGIC   RawSales a
# MAGIC   INNER JOIN qty_table b ON a.sales_main_key = b.sales_main_key
# MAGIC WHERE
# MAGIC   total_qty != 0
# MAGIC   AND order_date >= "2022-11-01" AND order_date <= "2023-10-31" 
# MAGIC   AND shop_brand = 'JB'

# COMMAND ----------

# MAGIC %md
# MAGIC ## product data

# COMMAND ----------

# MAGIC %sql
# MAGIC -- cleaned sku table
# MAGIC CREATE
# MAGIC OR REPLACE TEMPORARY VIEW SkuTable AS
# MAGIC WITH cleaned_sku AS (
# MAGIC SELECT
# MAGIC     DISTINCT item_code,
# MAGIC     item_desc,
# MAGIC     item_desc_c,
# MAGIC     item_cat,
# MAGIC     MAINCAT_KEY,
# MAGIC     item_sub_cat,
# MAGIC     item_product_line,
# MAGIC     item_product_line_desc,
# MAGIC     brand_code,
# MAGIC     retail_price_hk,
# MAGIC     retail_price_tw,
# MAGIC     CAST(last_modified_date AS DATE) AS last_modified_date
# MAGIC FROM
# MAGIC     imx_prd.imx_dw_train_silver.dbo_viw_lc_cs2k_item_sku
# MAGIC )
# MAGIC ,
# MAGIC SkuTable AS (
# MAGIC SELECT
# MAGIC     a.*
# MAGIC FROM
# MAGIC     cleaned_sku a
# MAGIC     INNER JOIN (
# MAGIC         SELECT
# MAGIC             item_code,
# MAGIC             MAX(last_modified_date) AS MaxDateTime
# MAGIC         FROM
# MAGIC             cleaned_sku
# MAGIC         GROUP BY
# MAGIC             item_code
# MAGIC     ) b ON a.item_code = b.item_code AND a.last_modified_date = b.MaxDateTime
# MAGIC ), 
# MAGIC cleaned_maincat AS (
# MAGIC   SELECT 
# MAGIC     DISTINCT maincat_key,
# MAGIC     maincat_code,
# MAGIC     maincat_desc
# MAGIC   FROM imx_prd.imx_dw_train_silver.dbo_viw_lc_cs2k_item_item_maincat
# MAGIC )
# MAGIC SELECT
# MAGIC   a.*,
# MAGIC   b.maincat_code,
# MAGIC   b.maincat_desc
# MAGIC from SkuTable a
# MAGIC left join cleaned_maincat b using (maincat_key)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- cleaned item_subcat table
# MAGIC CREATE
# MAGIC OR REPLACE TEMPORARY VIEW ItemSubcat AS WITH subcat AS (
# MAGIC   SELECT
# MAGIC     DISTINCT item_brand_key,
# MAGIC     item_cat,
# MAGIC     item_sub_cat,
# MAGIC     item_subcat_desc,
# MAGIC     load_date
# MAGIC   FROM
# MAGIC     imx_prd.imx_dw_train_silver.dbo_viw_lc_cs2k_item_item_subcat
# MAGIC )
# MAGIC SELECT
# MAGIC   DISTINCT a.item_sub_cat,
# MAGIC   item_cat,
# MAGIC   a.item_brand_key,
# MAGIC   a.item_subcat_desc
# MAGIC FROM
# MAGIC   subcat a
# MAGIC   INNER JOIN (
# MAGIC     SELECT
# MAGIC       item_brand_key,
# MAGIC       item_sub_cat,
# MAGIC       MAX(load_date) AS MaxDateTime
# MAGIC     FROM
# MAGIC       subcat
# MAGIC     GROUP BY
# MAGIC       item_brand_key,
# MAGIC       item_sub_cat
# MAGIC   ) b ON a.item_sub_cat = b.item_sub_cat
# MAGIC   AND a.load_date = b.MaxDateTime
# MAGIC   AND a.item_brand_key = b.item_brand_key
# MAGIC   -- AND item_cat NOT IN (
# MAGIC   --   "ZZ",
# MAGIC   --   "ZZZ"
# MAGIC   -- )

# COMMAND ----------

# MAGIC %sql
# MAGIC -- cleaned location table
# MAGIC CREATE
# MAGIC OR REPLACE TEMPORARY VIEW StoreLocation AS
# MAGIC WITH locn0 AS (
# MAGIC   SELECT
# MAGIC       shop_code,
# MAGIC       shop_desc,
# MAGIC       city_district,
# MAGIC       CAST(modified_date AS DATE) AS modified_date
# MAGIC   FROM
# MAGIC       imx_prd.imx_dw_train_silver.dbo_viw_lc_cs2k_inv_location
# MAGIC   WHERE
# MAGIC       shop_brand IN ('JB')
# MAGIC ),
# MAGIC locn1 AS (
# MAGIC   SELECT
# MAGIC       DISTINCT *
# MAGIC   FROM
# MAGIC       locn0
# MAGIC ),
# MAGIC locn2 AS (
# MAGIC   SELECT
# MAGIC       a.shop_code,
# MAGIC       a.shop_desc,
# MAGIC       a.city_district
# MAGIC   FROM
# MAGIC       locn1 a
# MAGIC       INNER JOIN (
# MAGIC           SELECT
# MAGIC               shop_code,
# MAGIC               MAX(modified_date) AS MaxDateTime
# MAGIC           FROM
# MAGIC               locn1
# MAGIC           GROUP BY
# MAGIC               shop_code
# MAGIC       ) b ON a.shop_code = b.shop_code
# MAGIC       AND a.modified_date = b.MaxDateTime
# MAGIC )
# MAGIC SELECT
# MAGIC     DISTINCT shop_code,
# MAGIC     shop_desc,
# MAGIC     city_district
# MAGIC FROM
# MAGIC     locn2;

# COMMAND ----------

# MAGIC %md
# MAGIC ## join product and sales (for sales table filtering)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- validate transaction filter, same for imx crm dashboard, provided by imx crm team
# MAGIC CREATE
# MAGIC OR REPLACE TEMPORARY VIEW JBSalesProduct AS 
# MAGIC WITH RawJBSalesProduct AS (
# MAGIC   SELECT
# MAGIC     DISTINCT a.*,
# MAGIC     b.item_desc,
# MAGIC     b.item_cat,
# MAGIC     b.item_sub_cat,
# MAGIC     b.brand_code,
# MAGIC     b.retail_price_hk,
# MAGIC     b.retail_price_tw,
# MAGIC     b.item_product_line_desc,
# MAGIC     b.maincat_desc,
# MAGIC     c.item_subcat_desc,
# MAGIC     d.shop_desc
# MAGIC   FROM
# MAGIC     JoyceBeautySales a
# MAGIC     LEFT JOIN SkuTable b ON a.item_code = b.item_code
# MAGIC     AND a.prod_brand = b.brand_code
# MAGIC     LEFT JOIN ItemSubcat c ON c.item_brand_key = a.prod_brand
# MAGIC     AND c.item_sub_cat = b.item_sub_cat
# MAGIC     AND c.item_cat = b.item_cat
# MAGIC     LEFT JOIN StoreLocation d USING (shop_code)
# MAGIC ), 
# MAGIC filtered AS (
# MAGIC SELECT 
# MAGIC   DISTINCT *
# MAGIC FROM RawJBSalesProduct
# MAGIC WHERE
# MAGIC   shop_brand = "JB"
# MAGIC   AND region_key = 'HK'
# MAGIC   AND item_code != "JBDUMITMJBY"
# MAGIC   AND order_date != "2021-02-27"
# MAGIC   AND valid_tx_flag = 1
# MAGIC   AND isnull(void_flag) = 1
# MAGIC )
# MAGIC SELECT
# MAGIC     a.*,
# MAGIC     b.item_subcat_desc AS item_subcat_desc_from_csv
# MAGIC FROM
# MAGIC     filtered a
# MAGIC     LEFT JOIN jb_subcat b ON a.item_sub_cat = b.item_sub_cat

# COMMAND ----------

# MAGIC %md
# MAGIC ## demographic data

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE
# MAGIC OR REPLACE TEMPORARY VIEW JBSalesVip AS
# MAGIC WITH cleaned_sales_vip AS (
# MAGIC SELECT 
# MAGIC     DISTINCT VIP_NO,
# MAGIC     MIN(VIP_MAIN_NO) VIP_MAIN_NO,
# MAGIC     VIP_TYPE,
# MAGIC     VIP_ISSUE_DATE,
# MAGIC     VIP_SEX,
# MAGIC     VIP_NATION,
# MAGIC     VIP_AGEGRP,
# MAGIC     REGION_KEY,
# MAGIC     VIP_STAFF_FLAG
# MAGIC FROM
# MAGIC     imx_prd.imx_dw_train_silver.dbo_viw_lc_sales_vip_masked
# MAGIC WHERE
# MAGIC     isnull(VIP_MAIN_NO) = 0
# MAGIC GROUP BY
# MAGIC     VIP_NO,
# MAGIC     VIP_TYPE,
# MAGIC     VIP_ISSUE_DATE,
# MAGIC     VIP_SEX,
# MAGIC     VIP_NATION,
# MAGIC     VIP_AGEGRP,
# MAGIC     REGION_KEY,
# MAGIC     VIP_STAFF_FLAG
# MAGIC )
# MAGIC SELECT  
# MAGIC   *
# MAGIC FROM cleaned_sales_vip

# COMMAND ----------

# MAGIC %md
# MAGIC ## first purchase

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE
# MAGIC OR REPLACE TEMPORARY VIEW first_purchase AS
# MAGIC WITH data0 AS (
# MAGIC SELECT
# MAGIC     CAST(TRIM(a.sales_date) AS DATE) AS order_date, -- sales_sales table is not trimmed
# MAGIC     TRIM(a.shop_brand) AS shop_brand,
# MAGIC     TRIM(a.vip_no) AS vip_no
# MAGIC FROM
# MAGIC     imx_prd.imx_dw_train_silver.dbo_viw_lc_sales_sales a
# MAGIC ),
# MAGIC data1 AS (
# MAGIC SELECT
# MAGIC     DISTINCT *
# MAGIC FROM
# MAGIC     data0
# MAGIC WHERE
# MAGIC     shop_brand IN ('JB')
# MAGIC ), 
# MAGIC sales0 AS (
# MAGIC SELECT
# MAGIC     a.*,
# MAGIC     COALESCE(b.vip_main_no, a.vip_no) vip_main_no
# MAGIC FROM
# MAGIC     data1 a
# MAGIC     LEFT JOIN vip b ON a.vip_no = b.vip_no
# MAGIC ),
# MAGIC sales AS (
# MAGIC SELECT
# MAGIC     DISTINCT vip_main_no,
# MAGIC     order_date,
# MAGIC     shop_brand
# MAGIC FROM
# MAGIC     sales0
# MAGIC ), 
# MAGIC first_pur0 AS (
# MAGIC SELECT
# MAGIC     vip_main_no,
# MAGIC     shop_brand,
# MAGIC     MIN(order_date) AS first_transaction_date
# MAGIC     -- MAX(order_date) AS last_transaction_date
# MAGIC FROM
# MAGIC     sales
# MAGIC GROUP BY
# MAGIC     vip_main_no,
# MAGIC     shop_brand
# MAGIC ), 
# MAGIC first_pur1 AS (
# MAGIC SELECT
# MAGIC     vip_main_no,
# MAGIC     CASE
# MAGIC         WHEN shop_brand = 'JB' THEN first_transaction_date
# MAGIC         ELSE NULL
# MAGIC     END AS first_pur_jb
# MAGIC FROM
# MAGIC     first_pur0
# MAGIC GROUP BY
# MAGIC     vip_main_no,
# MAGIC     shop_brand,
# MAGIC     first_transaction_date
# MAGIC )
# MAGIC SELECT
# MAGIC     vip_main_no,
# MAGIC     COALESCE(MIN(first_pur_jb), NULL) AS first_pur_jb
# MAGIC FROM
# MAGIC     first_pur1
# MAGIC GROUP BY
# MAGIC     vip_main_no

# COMMAND ----------

# DBTITLE 1,Output
# save to imx_dev (?)
# save as parquet for now
import os

base_dir = "/mnt/dev/customer_segmentation/imx/joyce_beauty/datamart"
os.makedirs(base_dir, exist_ok=True)

spark.table("JBSalesProduct").write.parquet(os.path.join(base_dir, "transaction.parquet"), mode="overwrite")
spark.table("JBSalesVip").write.parquet(os.path.join(base_dir, "demographic.parquet"), mode="overwrite")
spark.table("first_purchase").write.parquet(os.path.join(base_dir, "first_purchase.parquet"), mode="overwrite")

# COMMAND ----------


