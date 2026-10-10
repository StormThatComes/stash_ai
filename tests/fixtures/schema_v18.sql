-- Authentic v18 DDL from commit d24e566e92eb566411f10cf2eb7049b0ce598f5d.
-- Extracted by executing that commit's database/progress/archive/recovery/cleanup/merge-check CREATE statements in an empty SQLite database.
-- sqlite3.Connection.iterdump() records the resulting historical schema; no current initializers or renumbered current schema are used.
PRAGMA application_id=1398033735;
PRAGMA user_version=18;
BEGIN TRANSACTION;
CREATE TABLE checkpoint_analysis (
            scope TEXT NOT NULL, scene TEXT NOT NULL, model TEXT NOT NULL, data BLOB NOT NULL,
            PRIMARY KEY(scope,scene,model));
CREATE TABLE checkpoint_frames (
            scope TEXT NOT NULL, scene TEXT NOT NULL, model TEXT NOT NULL,
            processed INTEGER NOT NULL, data BLOB NOT NULL,
            PRIMARY KEY(scope,scene,model,processed));
CREATE TABLE checkpoints (
            scope TEXT NOT NULL, scene TEXT NOT NULL, model TEXT NOT NULL,
            signature TEXT NOT NULL, processed INTEGER NOT NULL, total INTEGER NOT NULL,
            top_rows INTEGER NOT NULL, top_scores BLOB NOT NULL,
            PRIMARY KEY(scope,scene,model));
CREATE TABLE cleanup_state (
            scope TEXT PRIMARY KEY, generation INTEGER NOT NULL, data BLOB NOT NULL,
            pending TEXT);
CREATE TABLE inference_archive (
        scope TEXT NOT NULL, scene TEXT NOT NULL, model TEXT NOT NULL,
        kind TEXT NOT NULL, signature TEXT NOT NULL, data BLOB NOT NULL,
        PRIMARY KEY(scope,scene,model,kind,signature));
CREATE TABLE merge_checks (
            scope TEXT NOT NULL, scene TEXT NOT NULL, policy TEXT NOT NULL,
            fingerprint TEXT NOT NULL, PRIMARY KEY(scope, scene));
CREATE TABLE recovery_cleanups (
            scope TEXT NOT NULL, generation INTEGER NOT NULL, kinds TEXT NOT NULL,
            PRIMARY KEY(scope,generation));
CREATE TABLE recovery_publications (
            scope TEXT NOT NULL, scene TEXT NOT NULL, kind TEXT NOT NULL,
            operation TEXT NOT NULL, data BLOB NOT NULL,
            PRIMARY KEY(scope,scene,kind,operation));
CREATE TABLE refresh_requests (
            scope TEXT NOT NULL, scene TEXT NOT NULL, model TEXT NOT NULL, kind TEXT NOT NULL,
            PRIMARY KEY(scope,scene,model,kind));
CREATE TABLE result_index (
            scope TEXT NOT NULL, scene TEXT NOT NULL, model TEXT NOT NULL, kind TEXT NOT NULL,
            inference INTEGER NOT NULL, populated INTEGER NOT NULL, signature TEXT,
            PRIMARY KEY(scope,scene,model,kind));
CREATE TABLE result_metadata (
                scope TEXT NOT NULL, scene TEXT NOT NULL, model TEXT NOT NULL,
                kind TEXT NOT NULL, format INTEGER NOT NULL, data TEXT NOT NULL,
                PRIMARY KEY(scope, scene, model, kind));
CREATE TABLE result_thresholds (
                scope TEXT NOT NULL, scene TEXT NOT NULL, model TEXT NOT NULL,
                kind TEXT NOT NULL, threshold REAL NOT NULL CHECK(typeof(threshold) IN ('real','integer') AND threshold BETWEEN 0 AND 1),
                PRIMARY KEY(scope, scene, model, kind));
CREATE TABLE results (
                scope TEXT NOT NULL, scene TEXT NOT NULL, model TEXT NOT NULL,
                kind TEXT NOT NULL, data BLOB NOT NULL,
                PRIMARY KEY(scope, scene, model, kind));
CREATE TABLE snapshots (
                scope TEXT NOT NULL, scene TEXT NOT NULL, kind TEXT NOT NULL,
                published BLOB, pending BLOB,
                PRIMARY KEY(scope, scene, kind));
CREATE TRIGGER results_delete_threshold
                AFTER DELETE ON results BEGIN
                    DELETE FROM result_thresholds WHERE scope=OLD.scope AND scene=OLD.scene
                        AND model=OLD.model AND kind=OLD.kind;
                END;
CREATE TRIGGER results_metadata_insert
                AFTER INSERT ON results BEGIN DELETE FROM result_metadata WHERE scope=NEW.scope AND scene=NEW.scene AND model=NEW.model AND kind=NEW.kind; END;
CREATE TRIGGER results_metadata_update
                AFTER UPDATE ON results BEGIN DELETE FROM result_metadata WHERE scope=OLD.scope AND scene=OLD.scene AND model=OLD.model AND kind=OLD.kind;DELETE FROM result_metadata WHERE scope=NEW.scope AND scene=NEW.scene AND model=NEW.model AND kind=NEW.kind; END;
CREATE TRIGGER results_metadata_delete
                AFTER DELETE ON results BEGIN DELETE FROM result_metadata WHERE scope=OLD.scope AND scene=OLD.scene AND model=OLD.model AND kind=OLD.kind; END;
CREATE INDEX snapshots_pending_scope
            ON snapshots(scope, scene, kind) WHERE pending IS NOT NULL;
CREATE TRIGGER perf_results_insert
                AFTER INSERT ON results BEGIN
                    DELETE FROM merge_checks WHERE scope=NEW.scope AND scene=NEW.scene;
                END;
CREATE TRIGGER perf_results_update
                AFTER UPDATE ON results BEGIN
                    DELETE FROM merge_checks WHERE scope=NEW.scope AND scene=NEW.scene;
                END;
CREATE TRIGGER perf_results_delete
                AFTER DELETE ON results BEGIN
                    DELETE FROM merge_checks WHERE scope=OLD.scope AND scene=OLD.scene;
                END;
CREATE TRIGGER perf_result_thresholds_insert
                AFTER INSERT ON result_thresholds BEGIN
                    DELETE FROM merge_checks WHERE scope=NEW.scope AND scene=NEW.scene;
                END;
CREATE TRIGGER perf_result_thresholds_update
                AFTER UPDATE ON result_thresholds BEGIN
                    DELETE FROM merge_checks WHERE scope=NEW.scope AND scene=NEW.scene;
                END;
CREATE TRIGGER perf_result_thresholds_delete
                AFTER DELETE ON result_thresholds BEGIN
                    DELETE FROM merge_checks WHERE scope=OLD.scope AND scene=OLD.scene;
                END;
CREATE TRIGGER perf_snapshots_insert
                AFTER INSERT ON snapshots BEGIN
                    DELETE FROM merge_checks WHERE scope=NEW.scope AND scene=NEW.scene;
                END;
CREATE TRIGGER perf_snapshots_update
                AFTER UPDATE ON snapshots BEGIN
                    DELETE FROM merge_checks WHERE scope=NEW.scope AND scene=NEW.scene;
                END;
CREATE TRIGGER perf_snapshots_delete
                AFTER DELETE ON snapshots BEGIN
                    DELETE FROM merge_checks WHERE scope=OLD.scope AND scene=OLD.scene;
                END;
CREATE TRIGGER perf_cleanup_insert
            AFTER INSERT ON cleanup_state BEGIN
                DELETE FROM merge_checks WHERE scope=NEW.scope;
            END;
CREATE TRIGGER perf_cleanup_update
            AFTER UPDATE ON cleanup_state BEGIN
                DELETE FROM merge_checks WHERE scope=NEW.scope;
            END;
CREATE TRIGGER perf_cleanup_delete
            AFTER DELETE ON cleanup_state BEGIN
                DELETE FROM merge_checks WHERE scope=OLD.scope;
            END;
COMMIT;
