from roboflow import Roboflow
import argparse
import json
from pathlib import Path


def extract_roboflow_details(url):
    """Parse Roboflow URL"""
    clean_url = url.split("?")[0]
    parts = clean_url.replace("https://", "").replace("http://", "").split("/")

    if len(parts) < 3:
        return None, None

    workspace = parts[1]
    project = parts[2]

    return workspace, project


def list_versions_for_url(rf, url):
    """List all available versions for a Roboflow dataset"""
    workspace, project_id = extract_roboflow_details(url)

    if not workspace or not project_id:
        print(f"⚠️  Invalid URL: {url}")
        return None

    try:
        print(f"\n{'='*80}")
        print(f"Dataset: {workspace}/{project_id}")
        print(f"URL: {url}")
        print(f"{'='*80}")

        project = rf.workspace(workspace).project(project_id)

        versions = project.versions()

        if not versions:
            print("No versions found")
            return None

        version_info = []

        for v in versions:
            info = {
                'version': v.version,
                'id': v.id,
                'name': v.name,
                'created': str(v.created) if hasattr(v, 'created') else 'N/A',
            }

            if hasattr(v, 'images'):
                info['num_images'] = v.images
            if hasattr(v, 'classes'):
                info['num_classes'] = len(v.classes) if v.classes else 0

            version_info.append(info)

            print(f"  Version {v.version}: {v.name}")
            if hasattr(v, 'created'):
                print(f"    Created: {v.created}")
            if hasattr(v, 'images'):
                print(f"    Images: {v.images}")
            if hasattr(v, 'classes') and v.classes:
                print(f"    Classes: {len(v.classes)}")

        latest = versions[0]
        print(f"\n  ✅ Latest version: {latest.version}")
        print(f"     Name: {latest.name}")

        return {
            'workspace': workspace,
            'project': project_id,
            'url': url,
            'versions': version_info,
            'latest_version': latest.version,
            'latest_url': f"https://universe.roboflow.com/{workspace}/{project_id}/model/{latest.version}"
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser("List Roboflow dataset versions")
    parser.add_argument('--urls', nargs='+', help='Roboflow URLs')
    parser.add_argument('--urls_file', type=str, help='File with URLs (one per line)')
    parser.add_argument('--api_key', type=str, required=True, help='Roboflow API key')
    parser.add_argument('--output_json', type=str, help='Save results to JSON')
    parser.add_argument('--create_config', action='store_true', help='Create dataset config with latest versions')

    args = parser.parse_args()

    rf = Roboflow(api_key=args.api_key)

    urls = []
    if args.urls:
        urls.extend(args.urls)

    if args.urls_file:
        with open(args.urls_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    urls.append(line)

    if not urls:
        print("Error: No URLs provided. Use --urls or --urls_file")
        return

    print(f"Checking {len(urls)} datasets...\n")

    all_results = []

    for url in urls:
        result = list_versions_for_url(rf, url)
        if result:
            all_results.append(result)

    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n📄 Results saved to: {args.output_json}")

    if args.create_config and all_results:
        config = []
        for result in all_results:
            config.append({
                'url': result['latest_url'],
                'category': result['project'],
                'min_objects': 2
            })

        config_path = 'dataset_configs_latest.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n📝 Dataset config created: {config_path}")
        print(f"\nYou can now run:")
        print(f"  python download_and_convert.py --config {config_path} --api_key YOUR_KEY")

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for result in all_results:
        print(f"{result['project']:40s} - Latest: v{result['latest_version']}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()


# python list_roboflow_versions.py \                                                                                                                                                                                                                                           first_version * ] 10:50 PM
#   --urls_file urls_list.txt \
#   --api_key iqi8O8ApF7TcURJLyVun \
#   --output_json versions_info.json \
#   --create_config

