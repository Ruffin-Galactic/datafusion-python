use std::fs;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::{Py, Python, PyResult};
use pyo3::exceptions::{PyValueError, PyRuntimeError, PyTimeoutError};

use datafusion::prelude::*;
use datafusion::execution::context::{SessionContext};
use datafusion::execution::runtime_env::{RuntimeEnv, RuntimeConfig};

use tokio::runtime::Runtime;
use arrow::pyarrow::IntoPyArrow;
use futures::executor::block_on;

use iceberg_sql_catalog::SqlCatalog;
use datafusion_iceberg::DataFusionTable;
use iceberg_rust::catalog::{Catalog};
use iceberg_rust::catalog::identifier::Identifier;
use iceberg_rust::catalog::tabular::Tabular;
use iceberg_rust::object_store::{ObjectStoreBuilder, ConfigKey};
use iceberg_rust::catalog::namespace::Namespace;
use crate::context::PySessionContext;
use iceberg_rust::table::Table;
use tokio::time::{timeout, Duration};
use datafusion::execution::disk_manager::DiskManagerConfig;



#[pyclass]
pub struct PyIcebergSessionContext {
    inner: SessionContext,
}

#[pymethods]
impl PyIcebergSessionContext {
    #[new]
    pub fn new(limit_bytes: Option<usize>) -> PyResult<Self> {
        let disk_manager = DiskManagerConfig::new();
        let runtime_env = Arc::new(
            RuntimeEnv::new(
                RuntimeConfig::new()
                    .with_memory_limit(limit_bytes.unwrap_or(8 * 1024 * 1024 * 1024), 1.0)
                    .with_disk_manager(disk_manager),
            )
            .map_err(|e| PyRuntimeError::new_err(format!("RuntimeEnv failed: {e}")))?,
        );

        let config = SessionConfig::new();
        let inner = SessionContext::new_with_config_rt(config, runtime_env);

        Ok(Self { inner })
    }
    
    

    pub fn register_iceberg_tables(
        &mut self,
        schema_name: &str,
        db_url: &str,
        catalog_name: &str,
        bucket: &str,
        service_account_path: Option<String>,
    ) -> PyResult<Py<PyAny>> {
        let bucket_key = "bucket".parse().map_err(|e| {
            PyValueError::new_err(format!("Invalid config key: {e}"))
        })?;

        let sa_key = "google_service_account_path".parse().map_err(|e| {
            PyValueError::new_err(format!("Invalid config key: {e}"))
        })?;

        let mut builder = ObjectStoreBuilder::gcs().with_config(bucket_key, bucket.to_string());

        if let Some(path) = service_account_path {
            builder = builder.with_config(sa_key, path);
        }

        let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to start Tokio runtime: {e}")))?;
        let _guard = rt.enter();
        let catalog_future = SqlCatalog::new(db_url, catalog_name, builder);
        let catalog = Arc::new(
            rt.block_on(timeout(Duration::from_secs(30), catalog_future))
                .map_err(|_| {
                    PyTimeoutError::new_err("Timeout: initializing catalog took too long")
                })?
                .map_err(|e| PyRuntimeError::new_err(format!("Catalog init failed: {e}")))?
        );

        let namespace = Namespace::try_new(&[schema_name.to_string()])
            .map_err(|e| PyValueError::new_err(format!("Invalid namespace: {e}")))?;

        let catalog_cloned = catalog.clone();
        let list_future = catalog_cloned.list_tabulars(&namespace);
        let tabulars = rt.block_on(timeout(Duration::from_secs(30), list_future))
            .map_err(|_| {
                PyTimeoutError::new_err(format!(
                    "Timeout: listing tables in schema '{schema_name}' took too long"
                ))
            })?
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to list tabulars: {e}")))?;

        let mut errors = vec![];

        for ident in &tabulars {
            let ident_str = ident.to_string();
            let catalog_cloned = catalog.clone();
            let load_future = catalog_cloned.load_tabular(ident);
            match rt.block_on(timeout(Duration::from_secs(30), load_future)) {
                Ok(Ok(tabular)) => {
                    if let Tabular::Table(t) = &tabular {
                        let table_provider = Arc::new(DataFusionTable::from(t.clone()));
                        let table_name = ident.name();
                        if let Err(e) = self.inner.register_table(table_name.to_string(), table_provider) {
                            errors.push(format!(" Failed to register table '{}': {}", table_name, e));
                        }
                    } else {
                        errors.push(format!(" Tabular '{}' is not a table", ident_str));
                    }
                }
                Ok(Err(e)) => {
                    let msg = format!("{e}");
                    if msg.contains("GCS") || msg.contains("decode") || msg.contains("credentials") {
                        errors.push(format!(" GCS credential or access error for '{}': {}", ident_str, msg));
                    } else {
                        errors.push(format!(" Failed to load tabular '{}': {}", ident_str, msg));
                    }
                }
                Err(_) => {
                    errors.push(format!(" Timeout: loading tabular '{}' took too long", ident_str));
                }
            }
        }

        if !errors.is_empty() {
            return Err(PyRuntimeError::new_err(errors.join("\n")));
        }

        Python::with_gil(|py| {
            let py_ctx_obj = Py::new(py, PySessionContext { ctx: self.inner.clone() })?;
            Ok(py_ctx_obj.to_object(py))
        })
    }

    pub fn sql(&self, py: Python, query: &str) -> PyResult<Py<PyAny>> {
        let rt = Runtime::new().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to start Tokio runtime: {e}"))
        })?;
        
        let df = rt.block_on(self.inner.sql(query)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Query failed: {e}"))
        })?;
        
        match rt.block_on(df.collect()) {
            Ok(batches) => {
                let no_valid_batches = batches.is_empty()
                || batches
                    .iter()
                    .all(|b| b.num_rows() == 0 || b.num_columns() == 0);

            if no_valid_batches {
                return Err(PyRuntimeError::new_err(
                    "Query returned no usable results (empty rows or incomplete schema).",
                ));
            }
                batches.into_pyarrow(py)
            }
            Err(e) => {
                let msg = format!("{e:?}");
                if msg.contains("ResourcesExhausted") || msg.contains("Additional allocation failed") {
                    Err(PyRuntimeError::new_err(" Query failed due to memory exhaustion. Try simplifying the query or increasing memory limits."))
                } else {
                    Err(PyRuntimeError::new_err(format!("Query execution failed: {e}")))
                }
            }
        }

    }
}
