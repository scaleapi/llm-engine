--
-- PostgreSQL database dump
--

-- Dumped from database version 13.12
-- Dumped by pg_dump version 13.16 (Ubuntu 13.16-1.pgdg20.04+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: hosted_model_inference; Type: SCHEMA; Schema: -; Owner: -
--

CREATE SCHEMA hosted_model_inference;


--
-- Name: model; Type: SCHEMA; Schema: -; Owner: -
--

CREATE SCHEMA model;


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: batch_jobs; Type: TABLE; Schema: hosted_model_inference; Owner: -
--

CREATE TABLE hosted_model_inference.batch_jobs (
    id text NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    completed_at timestamp with time zone,
    status text NOT NULL,
    created_by character varying(24) NOT NULL,
    owner character varying(24) NOT NULL,
    model_bundle_id text NOT NULL,
    model_endpoint_id text,
    task_ids_location text,
    result_location text
);


--
-- Name: bundles; Type: TABLE; Schema: hosted_model_inference; Owner: -
--

CREATE TABLE hosted_model_inference.bundles (
    id text NOT NULL,
    name character varying(50),
    created_by character varying(24),
    created_at timestamp with time zone DEFAULT now(),
    location text,
    version character varying(24),
    registered_model_name text,
    bundle_metadata json,
    requirements json,
    env_params json,
    packaging_type text,
    app_config json,
    model_artifact_ids text[] DEFAULT '{}'::text[],
    schema_location text,
    owner character varying(24) NOT NULL,
    flavor text,
    artifact_requirements text[],
    artifact_app_config json,
    artifact_framework_type text,
    artifact_pytorch_image_tag text,
    artifact_tensorflow_version text,
    artifact_image_repository text,
    artifact_image_tag text,
    cloudpickle_artifact_load_predict_fn text,
    cloudpickle_artifact_load_model_fn text,
    zip_artifact_load_predict_fn_module_path text,
    zip_artifact_load_model_fn_module_path text,
    runnable_image_repository text,
    runnable_image_tag text,
    runnable_image_command text[],
    runnable_image_env json,
    runnable_image_protocol text,
    artifact_location text,
    runnable_image_readiness_initial_delay_seconds integer,
    triton_enhanced_runnable_image_model_repository text,
    triton_enhanced_runnable_image_model_replicas json,
    triton_enhanced_runnable_image_num_cpu numeric,
    triton_enhanced_runnable_image_commit_tag text,
    triton_enhanced_runnable_image_storage text,
    triton_enhanced_runnable_image_memory text,
    triton_enhanced_runnable_image_readiness_initial_delay_seconds integer,
    streaming_enhanced_runnable_image_streaming_command text[],
    runnable_image_predict_route text,
    streaming_enhanced_runnable_image_streaming_predict_route text,
    runnable_image_healthcheck_route text,
    CONSTRAINT bundles_flavor_0 CHECK ((flavor = ANY (ARRAY['cloudpickle_artifact'::text, 'zip_artifact'::text, 'runnable_image'::text, 'triton_enhanced_runnable_image'::text, 'streaming_enhanced_runnable_image'::text]))),
    CONSTRAINT bundles_flavor_1 CHECK (((flavor ~~ '%_artifact'::text) = (artifact_requirements IS NOT NULL))),
    CONSTRAINT bundles_flavor_10 CHECK (((flavor = 'zip_artifact'::text) = (zip_artifact_load_predict_fn_module_path IS NOT NULL))),
    CONSTRAINT bundles_flavor_11 CHECK (((flavor = 'zip_artifact'::text) = (zip_artifact_load_model_fn_module_path IS NOT NULL))),
    CONSTRAINT bundles_flavor_12 CHECK (((flavor ~~ '%runnable_image'::text) = (runnable_image_repository IS NOT NULL))),
    CONSTRAINT bundles_flavor_13 CHECK (((flavor ~~ '%runnable_image'::text) = (runnable_image_tag IS NOT NULL))),
    CONSTRAINT bundles_flavor_14 CHECK (((flavor ~~ '%runnable_image'::text) = (runnable_image_command IS NOT NULL))),
    CONSTRAINT bundles_flavor_15 CHECK (((flavor ~~ '%runnable_image'::text) = (runnable_image_protocol IS NOT NULL))),
    CONSTRAINT bundles_flavor_16 CHECK (((flavor = 'triton_enhanced_runnable_image'::text) = (triton_enhanced_runnable_image_model_repository IS NOT NULL))),
    CONSTRAINT bundles_flavor_17 CHECK (((flavor = 'triton_enhanced_runnable_image'::text) = (triton_enhanced_runnable_image_num_cpu IS NOT NULL))),
    CONSTRAINT bundles_flavor_18 CHECK (((flavor = 'triton_enhanced_runnable_image'::text) = (triton_enhanced_runnable_image_commit_tag IS NOT NULL))),
    CONSTRAINT bundles_flavor_19 CHECK (((flavor = 'triton_enhanced_runnable_image'::text) = (triton_enhanced_runnable_image_readiness_initial_delay_seconds IS NOT NULL))),
    CONSTRAINT bundles_flavor_2 CHECK (((flavor ~~ '%_artifact'::text) = (artifact_location IS NOT NULL))),
    CONSTRAINT bundles_flavor_20 CHECK (((flavor = 'streaming_enhanced_runnable_image'::text) = (streaming_enhanced_runnable_image_streaming_command IS NOT NULL))),
    CONSTRAINT bundles_flavor_21 CHECK (((flavor ~~ '%runnable_image'::text) = (runnable_image_predict_route IS NOT NULL))),
    CONSTRAINT bundles_flavor_22 CHECK (((flavor ~~ '%runnable_image'::text) = (runnable_image_healthcheck_route IS NOT NULL))),
    CONSTRAINT bundles_flavor_23 CHECK (((flavor = 'streaming_enhanced_runnable_image'::text) = (streaming_enhanced_runnable_image_streaming_predict_route IS NOT NULL))),
    CONSTRAINT bundles_flavor_3 CHECK (((flavor ~~ '%_artifact'::text) = (artifact_framework_type IS NOT NULL))),
    CONSTRAINT bundles_flavor_4 CHECK (((artifact_framework_type = 'pytorch'::text) = (artifact_pytorch_image_tag IS NOT NULL))),
    CONSTRAINT bundles_flavor_5 CHECK (((artifact_framework_type = 'tensorflow'::text) = (artifact_tensorflow_version IS NOT NULL))),
    CONSTRAINT bundles_flavor_6 CHECK (((artifact_framework_type = 'custom_base_image'::text) = (artifact_image_repository IS NOT NULL))),
    CONSTRAINT bundles_flavor_7 CHECK (((artifact_framework_type = 'custom_base_image'::text) = (artifact_image_tag IS NOT NULL))),
    CONSTRAINT bundles_flavor_8 CHECK (((flavor = 'cloudpickle_artifact'::text) = (cloudpickle_artifact_load_predict_fn IS NOT NULL))),
    CONSTRAINT bundles_flavor_9 CHECK (((flavor = 'cloudpickle_artifact'::text) = (cloudpickle_artifact_load_model_fn IS NOT NULL)))
);


--
-- Name: docker_image_batch_job_bundles; Type: TABLE; Schema: hosted_model_inference; Owner: -
--

CREATE TABLE hosted_model_inference.docker_image_batch_job_bundles (
    id text NOT NULL,
    name text NOT NULL,
    created_by character varying(24) NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    owner character varying(24) NOT NULL,
    image_repository text NOT NULL,
    image_tag text NOT NULL,
    command text[] NOT NULL,
    env json NOT NULL,
    mount_location text,
    cpus text,
    memory text,
    storage text,
    gpus integer,
    gpu_type text,
    public boolean
);


--
-- Name: endpoints; Type: TABLE; Schema: hosted_model_inference; Owner: -
--

CREATE TABLE hosted_model_inference.endpoints (
    id text NOT NULL,
    name text,
    created_by character varying(24),
    created_at timestamp with time zone DEFAULT now(),
    last_updated_at timestamp with time zone DEFAULT now(),
    current_bundle_id text,
    endpoint_metadata jsonb,
    creation_task_id text,
    endpoint_type text,
    destination text,
    endpoint_status text,
    owner character varying(24) NOT NULL,
    public_inference boolean
);


--
-- Name: triggers; Type: TABLE; Schema: hosted_model_inference; Owner: -
--

CREATE TABLE hosted_model_inference.triggers (
    id character varying NOT NULL,
    name character varying NOT NULL,
    owner character varying NOT NULL,
    created_by character varying NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    cron_schedule character varying NOT NULL,
    docker_image_batch_job_bundle_id character varying NOT NULL,
    default_job_config jsonb,
    default_job_metadata jsonb
);


--
-- Name: model_artifacts; Type: TABLE; Schema: model; Owner: -
--

CREATE TABLE model.model_artifacts (
    id text NOT NULL,
    name text NOT NULL,
    description text,
    is_public boolean NOT NULL,
    created_by character varying(24) NOT NULL,
    owner character varying(24) NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    input_schema json,
    output_schema json,
    config json NOT NULL,
    location text NOT NULL,
    format text NOT NULL,
    format_metadata json NOT NULL,
    source text NOT NULL,
    source_metadata json NOT NULL
);


--
-- Name: model_versions; Type: TABLE; Schema: model; Owner: -
--

CREATE TABLE model.model_versions (
    id text NOT NULL,
    model_id text NOT NULL,
    version_number integer NOT NULL,
    tags text[] NOT NULL,
    created_by character varying(24) NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    launch_model_bundle_id text,
    nucleus_model_id text,
    metadata json DEFAULT '{}'::json NOT NULL
);


--
-- Name: models; Type: TABLE; Schema: model; Owner: -
--

CREATE TABLE model.models (
    id text NOT NULL,
    name text NOT NULL,
    description text,
    task_types text[] NOT NULL,
    created_by character varying(24) NOT NULL,
    owner character varying(24) NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: batch_jobs batch_jobs_pkey; Type: CONSTRAINT; Schema: hosted_model_inference; Owner: -
--

ALTER TABLE ONLY hosted_model_inference.batch_jobs
    ADD CONSTRAINT batch_jobs_pkey PRIMARY KEY (id);


--
-- Name: bundles bundles_pkey; Type: CONSTRAINT; Schema: hosted_model_inference; Owner: -
--

ALTER TABLE ONLY hosted_model_inference.bundles
    ADD CONSTRAINT bundles_pkey PRIMARY KEY (id);


--
-- Name: docker_image_batch_job_bundles docker_image_batch_job_bundles_pkey; Type: CONSTRAINT; Schema: hosted_model_inference; Owner: -
--

ALTER TABLE ONLY hosted_model_inference.docker_image_batch_job_bundles
    ADD CONSTRAINT docker_image_batch_job_bundles_pkey PRIMARY KEY (id);


--
-- Name: endpoints endpoint_name_created_by_uc; Type: CONSTRAINT; Schema: hosted_model_inference; Owner: -
--

ALTER TABLE ONLY hosted_model_inference.endpoints
    ADD CONSTRAINT endpoint_name_created_by_uc UNIQUE (name, created_by);


--
-- Name: endpoints endpoint_name_owner_uc; Type: CONSTRAINT; Schema: hosted_model_inference; Owner: -
--

ALTER TABLE ONLY hosted_model_inference.endpoints
    ADD CONSTRAINT endpoint_name_owner_uc UNIQUE (name, owner);


--
-- Name: endpoints endpoints_pkey; Type: CONSTRAINT; Schema: hosted_model_inference; Owner: -
--

ALTER TABLE ONLY hosted_model_inference.endpoints
    ADD CONSTRAINT endpoints_pkey PRIMARY KEY (id);


--
-- Name: triggers triggers_pkey; Type: CONSTRAINT; Schema: hosted_model_inference; Owner: -
--

ALTER TABLE ONLY hosted_model_inference.triggers
    ADD CONSTRAINT triggers_pkey PRIMARY KEY (id);


--
-- Name: triggers uq_triggers_name_owner; Type: CONSTRAINT; Schema: hosted_model_inference; Owner: -
--

ALTER TABLE ONLY hosted_model_inference.triggers
    ADD CONSTRAINT uq_triggers_name_owner UNIQUE (name, owner);


--
-- Name: model_versions launch_model_bundle_id_uc; Type: CONSTRAINT; Schema: model; Owner: -
--

ALTER TABLE ONLY model.model_versions
    ADD CONSTRAINT launch_model_bundle_id_uc UNIQUE (launch_model_bundle_id);


--
-- Name: model_artifacts model_artifacts_owner_name_uc; Type: CONSTRAINT; Schema: model; Owner: -
--

ALTER TABLE ONLY model.model_artifacts
    ADD CONSTRAINT model_artifacts_owner_name_uc UNIQUE (owner, name);


--
-- Name: model_artifacts model_artifacts_pkey; Type: CONSTRAINT; Schema: model; Owner: -
--

ALTER TABLE ONLY model.model_artifacts
    ADD CONSTRAINT model_artifacts_pkey PRIMARY KEY (id);


--
-- Name: model_versions model_id_version_number_uc; Type: CONSTRAINT; Schema: model; Owner: -
--

ALTER TABLE ONLY model.model_versions
    ADD CONSTRAINT model_id_version_number_uc UNIQUE (model_id, version_number);


--
-- Name: model_versions model_versions_pkey; Type: CONSTRAINT; Schema: model; Owner: -
--

ALTER TABLE ONLY model.model_versions
    ADD CONSTRAINT model_versions_pkey PRIMARY KEY (id);


--
-- Name: models models_owner_name_uc; Type: CONSTRAINT; Schema: model; Owner: -
--

ALTER TABLE ONLY model.models
    ADD CONSTRAINT models_owner_name_uc UNIQUE (owner, name);


--
-- Name: models models_pkey; Type: CONSTRAINT; Schema: model; Owner: -
--

ALTER TABLE ONLY model.models
    ADD CONSTRAINT models_pkey PRIMARY KEY (id);


--
-- Name: model_versions nucleus_model_id_uc; Type: CONSTRAINT; Schema: model; Owner: -
--

ALTER TABLE ONLY model.model_versions
    ADD CONSTRAINT nucleus_model_id_uc UNIQUE (nucleus_model_id);


--
-- Name: endpoint_name_llm_uc; Type: INDEX; Schema: hosted_model_inference; Owner: -
--

CREATE UNIQUE INDEX endpoint_name_llm_uc ON hosted_model_inference.endpoints USING btree (name) WHERE (endpoint_metadata ? '_llm'::text);


--
-- Name: idx_endpoint_metadata; Type: INDEX; Schema: hosted_model_inference; Owner: -
--

CREATE INDEX idx_endpoint_metadata ON hosted_model_inference.endpoints USING gin (endpoint_metadata);


--
-- Name: idx_trigger_name; Type: INDEX; Schema: hosted_model_inference; Owner: -
--

CREATE INDEX idx_trigger_name ON hosted_model_inference.triggers USING btree (name);


--
-- Name: ix_hosted_model_inference_batch_jobs_created_by; Type: INDEX; Schema: hosted_model_inference; Owner: -
--

CREATE INDEX ix_hosted_model_inference_batch_jobs_created_by ON hosted_model_inference.batch_jobs USING btree (created_by);


--
-- Name: ix_hosted_model_inference_batch_jobs_owner; Type: INDEX; Schema: hosted_model_inference; Owner: -
--

CREATE INDEX ix_hosted_model_inference_batch_jobs_owner ON hosted_model_inference.batch_jobs USING btree (owner);


--
-- Name: ix_hosted_model_inference_bundles_created_by; Type: INDEX; Schema: hosted_model_inference; Owner: -
--

CREATE INDEX ix_hosted_model_inference_bundles_created_by ON hosted_model_inference.bundles USING btree (created_by);


--
-- Name: ix_hosted_model_inference_bundles_name; Type: INDEX; Schema: hosted_model_inference; Owner: -
--

CREATE INDEX ix_hosted_model_inference_bundles_name ON hosted_model_inference.bundles USING btree (name);


--
-- Name: ix_hosted_model_inference_docker_image_batch_job_bundle_79a0; Type: INDEX; Schema: hosted_model_inference; Owner: -
--

CREATE INDEX ix_hosted_model_inference_docker_image_batch_job_bundle_79a0 ON hosted_model_inference.docker_image_batch_job_bundles USING btree (created_by);


--
-- Name: ix_hosted_model_inference_docker_image_batch_job_bundles_owner; Type: INDEX; Schema: hosted_model_inference; Owner: -
--

CREATE INDEX ix_hosted_model_inference_docker_image_batch_job_bundles_owner ON hosted_model_inference.docker_image_batch_job_bundles USING btree (owner);


--
-- Name: ix_hosted_model_inference_endpoints_created_by; Type: INDEX; Schema: hosted_model_inference; Owner: -
--

CREATE INDEX ix_hosted_model_inference_endpoints_created_by ON hosted_model_inference.endpoints USING btree (created_by);


--
-- Name: ix_hosted_model_inference_endpoints_name; Type: INDEX; Schema: hosted_model_inference; Owner: -
--

CREATE INDEX ix_hosted_model_inference_endpoints_name ON hosted_model_inference.endpoints USING btree (name);


--
-- Name: ix_model_model_artifacts_created_by; Type: INDEX; Schema: model; Owner: -
--

CREATE INDEX ix_model_model_artifacts_created_by ON model.model_artifacts USING btree (created_by);


--
-- Name: ix_model_model_artifacts_description; Type: INDEX; Schema: model; Owner: -
--

CREATE INDEX ix_model_model_artifacts_description ON model.model_artifacts USING btree (description);


--
-- Name: ix_model_model_artifacts_format; Type: INDEX; Schema: model; Owner: -
--

CREATE INDEX ix_model_model_artifacts_format ON model.model_artifacts USING btree (format);


--
-- Name: ix_model_model_artifacts_is_public; Type: INDEX; Schema: model; Owner: -
--

CREATE INDEX ix_model_model_artifacts_is_public ON model.model_artifacts USING btree (is_public);


--
-- Name: ix_model_model_artifacts_name; Type: INDEX; Schema: model; Owner: -
--

CREATE INDEX ix_model_model_artifacts_name ON model.model_artifacts USING btree (name);


--
-- Name: ix_model_model_artifacts_owner; Type: INDEX; Schema: model; Owner: -
--

CREATE INDEX ix_model_model_artifacts_owner ON model.model_artifacts USING btree (owner);


--
-- Name: ix_model_model_artifacts_source; Type: INDEX; Schema: model; Owner: -
--

CREATE INDEX ix_model_model_artifacts_source ON model.model_artifacts USING btree (source);


--
-- Name: ix_model_model_versions_created_by; Type: INDEX; Schema: model; Owner: -
--

CREATE INDEX ix_model_model_versions_created_by ON model.model_versions USING btree (created_by);


--
-- Name: ix_model_model_versions_model_id; Type: INDEX; Schema: model; Owner: -
--

CREATE INDEX ix_model_model_versions_model_id ON model.model_versions USING btree (model_id);


--
-- Name: ix_model_model_versions_tags; Type: INDEX; Schema: model; Owner: -
--

CREATE INDEX ix_model_model_versions_tags ON model.model_versions USING btree (tags);


--
-- Name: ix_model_model_versions_version_number; Type: INDEX; Schema: model; Owner: -
--

CREATE INDEX ix_model_model_versions_version_number ON model.model_versions USING btree (version_number);


--
-- Name: ix_model_models_created_by; Type: INDEX; Schema: model; Owner: -
--

CREATE INDEX ix_model_models_created_by ON model.models USING btree (created_by);


--
-- Name: ix_model_models_description; Type: INDEX; Schema: model; Owner: -
--

CREATE INDEX ix_model_models_description ON model.models USING btree (description);


--
-- Name: ix_model_models_name; Type: INDEX; Schema: model; Owner: -
--

CREATE INDEX ix_model_models_name ON model.models USING btree (name);


--
-- Name: ix_model_models_owner; Type: INDEX; Schema: model; Owner: -
--

CREATE INDEX ix_model_models_owner ON model.models USING btree (owner);


--
-- Name: ix_model_models_task_types; Type: INDEX; Schema: model; Owner: -
--

CREATE INDEX ix_model_models_task_types ON model.models USING btree (task_types);


--
-- Name: batch_jobs batch_jobs_model_bundle_id_fkey; Type: FK CONSTRAINT; Schema: hosted_model_inference; Owner: -
--

ALTER TABLE ONLY hosted_model_inference.batch_jobs
    ADD CONSTRAINT batch_jobs_model_bundle_id_fkey FOREIGN KEY (model_bundle_id) REFERENCES hosted_model_inference.bundles(id);


--
-- Name: batch_jobs batch_jobs_model_endpoint_id_fkey; Type: FK CONSTRAINT; Schema: hosted_model_inference; Owner: -
--

ALTER TABLE ONLY hosted_model_inference.batch_jobs
    ADD CONSTRAINT batch_jobs_model_endpoint_id_fkey FOREIGN KEY (model_endpoint_id) REFERENCES hosted_model_inference.endpoints(id) ON DELETE SET NULL;


--
-- Name: endpoints endpoints_current_bundle_id_fkey; Type: FK CONSTRAINT; Schema: hosted_model_inference; Owner: -
--

ALTER TABLE ONLY hosted_model_inference.endpoints
    ADD CONSTRAINT endpoints_current_bundle_id_fkey FOREIGN KEY (current_bundle_id) REFERENCES hosted_model_inference.bundles(id);


--
-- Name: triggers triggers_docker_image_batch_job_bundle_id_fkey; Type: FK CONSTRAINT; Schema: hosted_model_inference; Owner: -
--

ALTER TABLE ONLY hosted_model_inference.triggers
    ADD CONSTRAINT triggers_docker_image_batch_job_bundle_id_fkey FOREIGN KEY (docker_image_batch_job_bundle_id) REFERENCES hosted_model_inference.docker_image_batch_job_bundles(id);


--
-- Name: model_versions model_versions_launch_model_bundle_id_fkey; Type: FK CONSTRAINT; Schema: model; Owner: -
--

ALTER TABLE ONLY model.model_versions
    ADD CONSTRAINT model_versions_launch_model_bundle_id_fkey FOREIGN KEY (launch_model_bundle_id) REFERENCES hosted_model_inference.bundles(id);


--
-- Name: model_versions model_versions_model_id_fkey; Type: FK CONSTRAINT; Schema: model; Owner: -
--

ALTER TABLE ONLY model.model_versions
    ADD CONSTRAINT model_versions_model_id_fkey FOREIGN KEY (model_id) REFERENCES model.models(id);


--
-- Name: SCHEMA hosted_model_inference; Type: ACL; Schema: -; Owner: -
--

GRANT USAGE ON SCHEMA hosted_model_inference TO fivetran;


--
-- Name: SCHEMA model; Type: ACL; Schema: -; Owner: -
--

GRANT USAGE ON SCHEMA model TO fivetran;


--
-- Name: TABLE batch_jobs; Type: ACL; Schema: hosted_model_inference; Owner: -
--

GRANT SELECT ON TABLE hosted_model_inference.batch_jobs TO fivetran;


--
-- Name: TABLE bundles; Type: ACL; Schema: hosted_model_inference; Owner: -
--

GRANT SELECT ON TABLE hosted_model_inference.bundles TO fivetran;


--
-- Name: TABLE docker_image_batch_job_bundles; Type: ACL; Schema: hosted_model_inference; Owner: -
--

GRANT SELECT ON TABLE hosted_model_inference.docker_image_batch_job_bundles TO fivetran;


--
-- Name: TABLE endpoints; Type: ACL; Schema: hosted_model_inference; Owner: -
--

GRANT SELECT ON TABLE hosted_model_inference.endpoints TO fivetran;


--
-- Name: TABLE triggers; Type: ACL; Schema: hosted_model_inference; Owner: -
--

GRANT SELECT ON TABLE hosted_model_inference.triggers TO fivetran;


--
-- Name: TABLE model_artifacts; Type: ACL; Schema: model; Owner: -
--

GRANT SELECT ON TABLE model.model_artifacts TO fivetran;


--
-- Name: TABLE model_versions; Type: ACL; Schema: model; Owner: -
--

GRANT SELECT ON TABLE model.model_versions TO fivetran;


--
-- Name: TABLE models; Type: ACL; Schema: model; Owner: -
--

GRANT SELECT ON TABLE model.models TO fivetran;


--
-- Name: DEFAULT PRIVILEGES FOR TABLES; Type: DEFAULT ACL; Schema: hosted_model_inference; Owner: -
--

ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA hosted_model_inference GRANT SELECT ON TABLES  TO fivetran;


--
-- Name: DEFAULT PRIVILEGES FOR TABLES; Type: DEFAULT ACL; Schema: model; Owner: -
--

ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA model GRANT SELECT ON TABLES  TO fivetran;


--
-- PostgreSQL database dump complete
--

